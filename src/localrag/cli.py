"""Kommandozeile: ``localrag index``, ``search``, ``ask``, ``status``."""

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
from localrag.api.client import ServiceClient
from localrag.api.server import DEFAULT_HOST, DEFAULT_PORT
from localrag.config import DEFAULT_EXCLUDED_DIRS, DEFAULT_MAX_FILE_MB, ENV_DB, default_db_path
from localrag.generation.answer import AnswerResult, retrieve_sources, stream_answer
from localrag.generation.llm import ENV_LLM_MODEL, ENV_LLM_URL, ChatClient, LLMError
from localrag.generation.prompts import source_label
from localrag.index.db import Database
from localrag.index.embedder import Embedder, SemanticUnavailableError, load_embedder, semantic_available
from localrag.index.vector_store import EmbeddingModelMismatchError, VectorStore
from localrag.ingest.indexer import IndexStats, index_paths
from localrag.ingest.scanner import FileInfo
from localrag.retrieval.hybrid import search as run_search

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
LocalOption = Annotated[
    bool, typer.Option("--local", help="Laufenden Dienst (`localrag serve`) ignorieren und direkt suchen.")
]


@app.command()
def serve(
    db: DbOption,
    host: Annotated[str, typer.Option(help="Adresse. Nur 127.0.0.1 ist privat!")] = DEFAULT_HOST,
    port: Annotated[int, typer.Option(help="Port.")] = DEFAULT_PORT,
    no_model: Annotated[bool, typer.Option("--no-model", help="Ohne Embedding-Modell (nur BM25).")] = False,
) -> None:
    """Startet den lokalen Dienst: Modell bleibt geladen, `search`/`ask` antworten dann sofort."""
    import uvicorn

    from localrag.api.server import create_app

    if host not in ("127.0.0.1", "localhost", "::1"):
        console.print(
            f"[bold red]Achtung:[/bold red] Mit --host {host} sind Inhalte deiner Dateien im Netzwerk lesbar."
        )
    if not db.exists():
        console.print(f"[red]Kein Index unter {db}[/red] – zuerst `localrag index <ordner>` ausführen.")
        raise typer.Exit(1)
    with console.status("Lade Embedding-Modell…"):
        application = create_app(db, load_model=not no_model)
    model = application.state.embedder.name if application.state.embedder else "keins (nur BM25)"
    console.print(f"[green]LocalRAG-Dienst[/green] auf http://{host}:{port} · Modell: {model} · Index: {db}")
    console.print("[dim]Beenden mit Strg+C[/dim]")
    uvicorn.run(application, host=host, port=port, log_level="warning")


@app.command()
def index(
    paths: Annotated[list[Path], typer.Argument(help="Ordner oder Dateien, die indexiert werden.")],
    db: DbOption,
    exclude: Annotated[
        list[str] | None, typer.Option("--exclude", "-x", help="Zusätzliche Ordnernamen ausschließen.")
    ] = None,
    max_file_mb: Annotated[float, typer.Option(help="Größere Dateien überspringen.")] = DEFAULT_MAX_FILE_MB,
    embed: Annotated[
        bool | None,
        typer.Option("--embed/--no-embed", help="Vektoren erzeugen (Standard: wenn installiert)."),
    ] = None,
    reembed: Annotated[
        bool, typer.Option(help="Alle Vektoren neu berechnen (z. B. nach Modellwechsel).")
    ] = False,
    model: Annotated[str | None, typer.Option(help="Embedding-Modell, Standard: BAAI/bge-m3.")] = None,
) -> None:
    """Indexiert Ordner inkrementell (nur neue und geänderte Dateien) und erzeugt Vektoren."""
    missing = [p for p in paths if not p.exists()]
    if missing:
        console.print(f"[red]Nicht gefunden:[/red] {', '.join(map(str, missing))}")
        raise typer.Exit(1)

    excluded = set(DEFAULT_EXCLUDED_DIRS) | set(exclude or [])
    embedder = _load_embedder_for_index(embed, model)
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

        embed_task = None

        def on_embed(n: int, total: int) -> None:
            nonlocal embed_task
            if embed_task is None:
                progress.update(task, visible=False)
                embed_task = progress.add_task("Vektoren", total=total)
            progress.update(embed_task, advance=n)

        try:
            stats = index_paths(
                database,
                paths,
                progress=on_file,
                excluded_dirs=excluded,
                max_file_mb=max_file_mb,
                embedder=embedder,
                reembed=reembed,
                embed_progress=on_embed,
            )
        except EmbeddingModelMismatchError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc

    elapsed = time.perf_counter() - started
    console.print(
        f"[green]Fertig[/green] in {elapsed:.1f}s · {stats.scanned} Dateien gescannt · "
        f"{stats.added} neu · {stats.updated} aktualisiert · {stats.unchanged} unverändert · "
        f"{stats.removed} entfernt · {stats.chunks} Chunks"
    )
    _print_embed_info(stats, embedder)
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
    mode: Annotated[
        str,
        typer.Option(
            "--mode", "-m", help="auto | keyword | semantic | hybrid (auto = hybrid, falls möglich)"
        ),
    ] = "auto",
    local: LocalOption = False,
) -> None:
    """Durchsucht den Index: Stichworte (BM25), Bedeutung (Vektoren) oder beides (hybrid)."""
    if not db.exists():
        console.print(f"[red]Kein Index unter {db}[/red] – zuerst `localrag index <ordner>` ausführen.")
        raise typer.Exit(1)
    if mode not in ("auto", "keyword", "semantic", "hybrid"):
        console.print(f"[red]Unbekannter Modus {mode!r}[/red]")
        raise typer.Exit(2)

    service = None if local else _service_for(db)
    if service is not None:
        try:
            with service:
                resolved, elapsed_ms, hits = service.search(
                    query, mode=mode, limit=limit, filetypes=filetype, chunks=all_chunks
                )
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        resolved += ", Dienst"
    else:
        with Database(db) as database:
            embedder = _load_embedder_for_search(database, mode)
            resolved = "keyword" if embedder is None else ("hybrid" if mode == "auto" else mode)
            started = time.perf_counter()
            try:
                hits = run_search(
                    database,
                    query,
                    mode=resolved,  # type: ignore[arg-type]
                    embedder=embedder,
                    limit=limit,
                    filetypes=filetype,
                    one_per_document=not all_chunks,
                )
            except EmbeddingModelMismatchError as exc:
                console.print(f"[red]{exc}[/red]")
                raise typer.Exit(1) from exc
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
    console.print(f"[dim]{len(hits)} Treffer in {elapsed_ms:.0f} ms ({resolved})[/dim]")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Frage in natürlicher Sprache.")],
    db: DbOption,
    k: Annotated[int, typer.Option("--sources", "-k", help="Maximale Anzahl Textstellen als Kontext.")] = 6,
    mode: Annotated[
        str, typer.Option("--mode", "-m", help="Suchmodus: auto | keyword | semantic | hybrid")
    ] = "auto",
    filetype: Annotated[list[str] | None, typer.Option("--type", "-t", help="Nur diese Dateitypen.")] = None,
    llm_url: Annotated[
        str | None, typer.Option("--llm-url", envvar=ENV_LLM_URL, help="OpenAI-kompatibler Server.")
    ] = None,
    llm_model: Annotated[
        str | None, typer.Option("--llm-model", envvar=ENV_LLM_MODEL, help="Modell-ID, z. B. qwen3.5-9b.")
    ] = None,
    show_context: Annotated[
        bool, typer.Option("--show-context", help="Alle übergebenen Textstellen zeigen.")
    ] = False,
    local: LocalOption = False,
) -> None:
    """Beantwortet eine Frage aus deinen Dateien – mit Quellenangaben (lokales LLM, z. B. LM Studio)."""
    if not db.exists():
        console.print(f"[red]Kein Index unter {db}[/red] – zuerst `localrag index <ordner>` ausführen.")
        raise typer.Exit(1)
    if mode not in ("auto", "keyword", "semantic", "hybrid"):
        console.print(f"[red]Unbekannter Modus {mode!r}[/red]")
        raise typer.Exit(2)

    # Der Dienst nutzt seine eigene LLM-Konfiguration → bei explizitem --llm-url lokal arbeiten
    service = None if (local or llm_url) else _service_for(db)
    if service is not None:
        with service:
            result, resolved, search_ms, answer_s, model_name = _ask_via_service(
                service,
                question,
                k=k,
                mode=mode,
                filetypes=filetype,
                llm_model=llm_model,
                show_context=show_context,
            )
        _print_answer_footer(result, f"{resolved}, Dienst", search_ms, answer_s, model_name)
        return

    with Database(db) as database, ChatClient(llm_url, llm_model) as client:
        try:
            model_name = client.model  # früh prüfen, ob der Server läuft
        except LLMError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc

        embedder = _load_embedder_for_search(database, mode)
        resolved = "keyword" if embedder is None else ("hybrid" if mode == "auto" else mode)
        started = time.perf_counter()
        sources = retrieve_sources(
            database,
            question,
            embedder=embedder,
            mode=resolved,
            k=k,
            filetypes=filetype,  # type: ignore[arg-type]
        )
        search_ms = (time.perf_counter() - started) * 1000
        result = AnswerResult(question=question, sources=sources)

        if show_context:
            for i, h in enumerate(sources, start=1):
                console.print(f"[dim][{i}] {source_label(h)}[/dim]")

        started = time.perf_counter()
        try:
            with console.status(f"{model_name} denkt nach…") as status:
                first = True
                for piece in stream_answer(client, result):
                    if first:
                        status.stop()
                        first = False
                    console.print(piece, end="", markup=False, highlight=False, soft_wrap=True)
        except LLMError as exc:
            console.print(f"\n[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        answer_s = time.perf_counter() - started
    _print_answer_footer(result, resolved, search_ms, answer_s, model_name)


def _ask_via_service(
    service: ServiceClient,
    question: str,
    *,
    k: int,
    mode: str,
    filetypes: list[str] | None,
    llm_model: str | None,
    show_context: bool,
) -> tuple[AnswerResult, str, float, float, str]:
    result = AnswerResult(question=question, sources=[])
    resolved, search_ms, model_name = mode, 0.0, "?"
    started = time.perf_counter()
    parts: list[str] = []
    try:
        with console.status("Suche und frage das Modell…") as status:
            for event in service.ask(question, k=k, mode=mode, filetypes=filetypes, llm_model=llm_model):
                kind = event["type"]
                if kind == "sources":
                    result.sources = event["sources"]
                    resolved, search_ms = event["mode"], event["search_ms"]
                    if show_context:
                        for i, h in enumerate(result.sources, start=1):
                            console.print(f"[dim][{i}] {source_label(h)}[/dim]")
                elif kind == "model":
                    model_name = event["model"] or "–"
                elif kind == "token":
                    status.stop()
                    parts.append(event["text"])
                    console.print(event["text"], end="", markup=False, highlight=False, soft_wrap=True)
                elif kind == "done":
                    result.cited = event["cited"]
                elif kind == "error":
                    status.stop()
                    console.print(f"\n[red]{event['message']}[/red]")
                    raise typer.Exit(1)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    result.text = "".join(parts).strip()
    return result, resolved, search_ms, time.perf_counter() - started, model_name


def _print_answer_footer(
    result: AnswerResult, resolved: str, search_ms: float, answer_s: float, model_name: str
) -> None:
    sources = result.sources
    console.print()
    if result.cited_sources:
        console.print("\n[bold]Quellen[/bold]")
        for n, h in result.cited_sources:
            location = f"  · Seite {h.page}" if h.page else ""
            if not location and h.heading_path:
                location = f"  · {h.heading_path}"
            console.print(Text.assemble((f"[{n}] ", "bold"), (h.path, "cyan"), (location, "dim")))
    elif sources and not result.is_no_answer:
        console.print(
            "\n[yellow]Die Antwort enthält keine Quellenangabe – bitte mit Vorsicht lesen.[/yellow]"
        )
    console.print(
        f"[dim]Suche {search_ms:.0f} ms ({resolved}, {len(sources)} Textstellen) · "
        f"Antwort {answer_s:.1f} s ({model_name})[/dim]"
    )


def _service_for(db: Path) -> ServiceClient | None:
    """Laufenden Dienst für genau diesen Index finden, sonst None (dann wird lokal gesucht)."""
    client = ServiceClient()
    if client.serves(db):
        return client
    client.close()
    return None


def _load_embedder_for_index(embed: bool | None, model: str | None) -> Embedder | None:
    if embed is False:
        return None
    if model and model.startswith("hashing-"):
        return load_embedder(model)
    if not semantic_available():
        if embed:
            console.print("[red]Semantische Suche nicht installiert:[/red] uv sync --extra semantic")
            raise typer.Exit(1)
        return None
    with console.status("Lade Embedding-Modell… (beim ersten Mal Download ~2,3 GB)"):
        try:
            return load_embedder(model)
        except SemanticUnavailableError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc


def _load_embedder_for_search(database: Database, mode: str) -> Embedder | None:
    if mode == "keyword":
        return None
    store = VectorStore(database)
    if not store.available or not store.model:
        if mode in ("semantic", "hybrid"):
            console.print("[red]Index hat keine Vektoren.[/red] Zuerst `localrag index <ordner> --embed`.")
            raise typer.Exit(1)
        return None
    if not store.model.startswith("hashing-") and not semantic_available():
        if mode in ("semantic", "hybrid"):
            console.print("[red]Semantische Suche nicht installiert:[/red] uv sync --extra semantic")
            raise typer.Exit(1)
        return None
    with console.status(f"Lade {store.model}…"):
        return load_embedder(store.model)


def _print_embed_info(stats: IndexStats, embedder: Embedder | None) -> None:
    if embedder is None:
        console.print("[dim]Ohne Vektoren (nur Stichwortsuche). Für semantische Suche: --embed[/dim]")
    else:
        console.print(f"[dim]{stats.embedded} neue Vektoren mit {embedder.name}[/dim]")


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
        store = VectorStore(database)
        vec_info = (
            f"Vektoren: {st['chunks'] - store.pending_count()}/{st['chunks']} ({store.model})"
            if store.available
            else "Vektoren: keine (nur Stichwortsuche)"
        )
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
    console.print(vec_info)
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
