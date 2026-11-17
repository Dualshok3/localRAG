"""Kommandozeile: ``localrag index``, ``localrag search``, ``localrag status``."""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from localrag import __version__
from localrag.config import DEFAULT_EXCLUDED_DIRS, DEFAULT_MAX_FILE_MB, ENV_DB, default_db_path
from localrag.index.db import Database
from localrag.ingest.indexer import index_paths
from localrag.ingest.scanner import FileInfo
from localrag.retrieval.keyword import keyword_search

# Windows-Konsolen nutzen teils noch cp1252 → Umlaute/Sonderzeichen nie mit Absturz quittieren.
for _stream in (sys.stdout, sys.stderr):
    if isinstance(_stream, io.TextIOWrapper):
        _stream.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(help="LocalRAG – lokale Suche über die eigenen Dateien.", no_args_is_help=True)
console = Console()

DbOption = Annotated[
    Path,
    typer.Option("--db", envvar=ENV_DB, help="Pfad zur Index-Datenbank.", default_factory=default_db_path),
]


@app.command()
def index(
    paths: Annotated[list[Path], typer.Argument(help="Ordner oder Dateien, die indexiert werden.")],
    db: DbOption,
    exclude: Annotated[
        list[str] | None, typer.Option("--exclude", "-x", help="Zusätzliche Ordnernamen ausschließen.")
    ] = None,
    max_file_mb: Annotated[float, typer.Option(help="Größere Dateien überspringen.")] = DEFAULT_MAX_FILE_MB,
) -> None:
    """Indexiert Ordner inkrementell (nur neue und geänderte Dateien)."""
    missing = [p for p in paths if not p.exists()]
    if missing:
        console.print(f"[red]Nicht gefunden:[/red] {', '.join(map(str, missing))}")
        raise typer.Exit(1)

    excluded = set(DEFAULT_EXCLUDED_DIRS) | set(exclude or [])
    started = time.perf_counter()
    with (
        Database(db) as database,
        Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress,
    ):
        task = progress.add_task("Indexiere…", total=None)

        def on_file(info: FileInfo, status: str) -> None:
            progress.update(task, advance=1, description=f"{status:<9} {info.path.name[:50]}")

        stats = index_paths(
            database, paths, progress=on_file, excluded_dirs=excluded, max_file_mb=max_file_mb
        )

    elapsed = time.perf_counter() - started
    console.print(
        f"[green]Fertig[/green] in {elapsed:.1f}s · {stats.scanned} Dateien gescannt · "
        f"{stats.added} neu · {stats.updated} aktualisiert · {stats.unchanged} unverändert · "
        f"{stats.removed} entfernt · {stats.chunks} Chunks"
    )
    if stats.empty:
        console.print(f"[yellow]{stats.empty} Dateien ohne Text[/yellow] (z. B. gescannte PDFs → OCR folgt)")
    if stats.failed:
        console.print(f"[red]{stats.failed} Fehler[/red] – Details mit `localrag status --errors`")
    console.print(f"[dim]Index: {db}[/dim]")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Suchanfrage in natürlicher Sprache oder Stichworte.")],
    db: DbOption,
    limit: Annotated[int, typer.Option("--limit", "-k", help="Anzahl Treffer.")] = 10,
    filetype: Annotated[
        list[str] | None, typer.Option("--type", "-t", help="Nur diese Dateitypen, z. B. -t pdf -t docx.")
    ] = None,
    all_chunks: Annotated[bool, typer.Option("--chunks", help="Mehrere Treffer pro Datei zeigen.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Ausgabe als JSON.")] = False,
) -> None:
    """Durchsucht den Index (aktuell: BM25-Stichwortsuche)."""
    if not db.exists():
        console.print(f"[red]Kein Index unter {db}[/red] – zuerst `localrag index <ordner>` ausführen.")
        raise typer.Exit(1)

    started = time.perf_counter()
    with Database(db) as database:
        hits = keyword_search(
            database, query, limit=limit, filetypes=filetype, one_per_document=not all_chunks
        )
    elapsed_ms = (time.perf_counter() - started) * 1000

    if as_json:
        payload = [
            {
                "path": h.path,
                "page": h.page,
                "heading": h.heading_path,
                "score": h.score,
                "snippet": h.snippet,
            }
            for h in hits
        ]
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if not hits:
        console.print(f"Keine Treffer ({elapsed_ms:.0f} ms).")
        return

    for i, h in enumerate(hits, start=1):
        location = f"Seite {h.page}" if h.page else (h.heading_path or "")
        header = Text.assemble((f"{i:>2}. ", "bold"), (h.path, "cyan"))
        if location:
            header.append(f"  · {location}", style="dim")
        console.print(header)
        snippet = Text("    " + h.snippet)
        snippet.highlight_words(h.matched_words, style="bold yellow", case_sensitive=False)
        console.print(snippet)
    console.print(f"[dim]{len(hits)} Treffer in {elapsed_ms:.0f} ms (BM25)[/dim]")


@app.command()
def status(
    db: DbOption,
    errors: Annotated[bool, typer.Option("--errors", help="Fehlerhafte Dateien auflisten.")] = False,
) -> None:
    """Zeigt Statistiken über den Index."""
    if not db.exists():
        console.print(f"Noch kein Index unter {db}.")
        raise typer.Exit(0)
    with Database(db) as database:
        st = database.stats()
        error_rows = database.errors() if errors else []

    by_status: dict[str, int] = st["by_status"]  # type: ignore[assignment]
    by_type: dict[str, int] = st["by_type"]  # type: ignore[assignment]
    size_mb = db.stat().st_size / 1024 / 1024

    console.print(f"[bold]LocalRAG {__version__}[/bold] · [dim]{db} ({size_mb:.1f} MB)[/dim]")
    table = Table()
    table.add_column("Dateityp")
    table.add_column("Dateien", justify="right")
    for ft, n in by_type.items():
        table.add_row(ft, str(n))
    console.print(table)
    console.print(
        f"Dokumente: {sum(by_status.values())} (ok {by_status.get('ok', 0)}, ohne Text "
        f"{by_status.get('empty', 0)}, Fehler {by_status.get('error', 0)}) · Chunks: {st['chunks']}"
    )
    for path, err in error_rows:
        console.print(f"[red]✗[/red] {path}\n   [dim]{err}[/dim]")


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", help="Version anzeigen.")] = False,
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


if __name__ == "__main__":
    app()
