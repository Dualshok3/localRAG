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
from localrag.api.server import DEFAULT_HOST, DEFAULT_PORT, LOCAL_HOSTS
from localrag.config import (
    CONFIG_TEMPLATE,
    DEFAULT_EXCLUDED_DIRS,
    ENV_DB,
    Config,
    ConfigError,
    config_path,
    default_db_path,
    load_config,
)
from localrag.generation.answer import AnswerResult, retrieve_sources, stream_answer
from localrag.generation.llm import ENV_LLM_MODEL, ENV_LLM_URL, ChatClient, LLMError
from localrag.generation.prompts import source_label
from localrag.index.db import Database
from localrag.index.embedder import Embedder, SemanticUnavailableError, load_embedder, semantic_available
from localrag.index.vector_store import EmbeddingModelMismatchError, VectorStore
from localrag.ingest.indexer import IndexStats, index_paths
from localrag.ingest.ocr import OcrUnavailableError, load_ocr
from localrag.ingest.scanner import FileInfo
from localrag.retrieval.filters import parse_query
from localrag.retrieval.hybrid import search as run_search
from localrag.retrieval.keyword import page_label
from localrag.retrieval.reranker import CrossEncoderReranker, Reranker

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


RerankOption = Annotated[
    bool,
    typer.Option(
        "--rerank",
        help="Treffer mit einem Cross-Encoder (bge-reranker-v2-m3) neu sortieren: besser bei Umschreibungen, "
        "aber ~0,3 s langsamer.",
    ),
]


def _load_reranker(rerank: bool) -> Reranker | None:
    if not rerank:
        return None
    try:
        with console.status("Lade Reranker…"):
            return CrossEncoderReranker()
    except SemanticUnavailableError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


@app.command()
def serve(
    db: DbOption,
    host: Annotated[str, typer.Option(help="Adresse. Nur 127.0.0.1 ist privat!")] = DEFAULT_HOST,
    port: Annotated[int, typer.Option(help="Port.")] = DEFAULT_PORT,
    no_model: Annotated[bool, typer.Option("--no-model", help="Ohne Embedding-Modell (nur BM25).")] = False,
    rerank: RerankOption = False,
    watch: Annotated[
        bool,
        typer.Option(
            "--watch",
            help="Zusätzlich die Ordner aus der Konfiguration beobachten und den Index aktuell halten.",
        ),
    ] = False,
) -> None:
    """Startet den lokalen Dienst: Modell bleibt geladen, `search`/`ask` antworten dann sofort."""
    import uvicorn

    from localrag.api.server import create_app
    from localrag.index.vector_store import enable_memory_cache

    enable_memory_cache()  # lang laufender Prozess → Vektoren im Arbeitsspeicher halten
    if host not in ("127.0.0.1", "localhost", "::1"):
        console.print(
            f"[bold red]Achtung:[/bold red] Mit --host {host} sind Inhalte deiner Dateien im Netzwerk lesbar."
        )
    if not db.exists():
        console.print(f"[red]Kein Index unter {db}[/red] – zuerst `localrag index <ordner>` ausführen.")
        raise typer.Exit(1)
    with console.status("Lade Embedding-Modell…"):
        cfg = _config_or_exit()
        application = create_app(
            db,
            load_model=not no_model,
            reranker=_load_reranker(rerank),
            allowed_hosts=(*LOCAL_HOSTS, host),
            inbox=cfg.inbox,
            extensions=cfg.extensions(),
            exclusions=cfg.exclusions(),
            max_file_mb=cfg.max_file_mb,
        )
    model = application.state.embedder.name if application.state.embedder else "keins (nur BM25)"
    if application.state.reranker:
        model += f" + {application.state.reranker.name}"
    console.print(f"[green]LocalRAG-Dienst[/green] auf http://{host}:{port} · Modell: {model} · Index: {db}")
    console.print(f"Web-Oberfläche: [link=http://{host}:{port}/]http://{host}:{port}/[/link]")
    watcher = None
    if watch:
        cfg = _config_or_exit()
        if not cfg.folders:
            console.print(f"[red]--watch braucht `folders` in {config_path()}[/red]")
            raise typer.Exit(2)
        watcher = _make_watcher(
            db, list(cfg.folders), cfg, None, application.state.embedder, application.state.model_lock
        )
        application.state.watcher = watcher
        watcher.start(catch_up=True)
        console.print("Beobachte: " + ", ".join(map(str, watcher.roots)))
    console.print("[dim]Beenden mit Strg+C[/dim]")
    try:
        uvicorn.run(application, host=host, port=port, log_level="warning")
    finally:
        if watcher is not None:
            watcher.stop()


@app.command("mcp")
def mcp_command(
    db: DbOption,
    allow: Annotated[
        list[Path] | None,
        typer.Option("--allow", help="Nur Treffer aus diesen Ordnern herausgeben (mehrfach möglich)."),
    ] = None,
    no_model: Annotated[bool, typer.Option("--no-model", help="Ohne Embedding-Modell (nur BM25).")] = False,
) -> None:
    """MCP-Server (stdio) für Claude Desktop / Claude Code: Suche in deinen Dateien als Werkzeug."""
    from localrag.api.server import default_embedder_for
    from localrag.index.vector_store import enable_memory_cache
    from localrag.mcp_server import FileSearchTools, build_server

    enable_memory_cache()

    # stdout gehört dem MCP-Protokoll → keine Konsolenausgaben hier
    tools = FileSearchTools(
        db,
        allowed_roots=allow or [],
        embedder_loader=None if no_model else (lambda: default_embedder_for(db)),
    )
    try:
        server = build_server(tools)
    except RuntimeError as exc:
        Console(stderr=True).print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    server.run()


@app.command()
def index(
    db: DbOption,
    paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Ordner oder Dateien. Ohne Angabe: `folders` aus der Konfiguration."),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option("--exclude", "-x", help="Zusätzlich ausschließen: Name, Muster (*backup*) oder Pfad."),
    ] = None,
    max_file_mb: Annotated[
        float | None, typer.Option(help="Größere Dateien überspringen (Standard aus Konfiguration: 50).")
    ] = None,
    embed: Annotated[
        bool | None,
        typer.Option("--embed/--no-embed", help="Vektoren erzeugen (Standard: wenn installiert)."),
    ] = None,
    reembed: Annotated[
        bool, typer.Option(help="Alle Vektoren neu berechnen (z. B. nach Modellwechsel).")
    ] = False,
    model: Annotated[str | None, typer.Option(help="Embedding-Modell, Standard: BAAI/bge-m3.")] = None,
    ocr: Annotated[
        str, typer.Option(help="Texterkennung für gescannte PDFs: auto, windows oder none.")
    ] = "auto",
) -> None:
    """Indexiert Ordner inkrementell (nur neue und geänderte Dateien) und erzeugt Vektoren."""
    cfg = _config_or_exit()
    paths = paths or list(cfg.folders)
    if not paths:
        console.print(
            "[red]Keine Ordner angegeben.[/red] `localrag index <ordner>` oder `folders` in "
            f"{config_path()} eintragen (Vorlage: `localrag config --init`)."
        )
        raise typer.Exit(2)
    missing = [p for p in paths if not p.exists()]
    if missing:
        console.print(f"[red]Nicht gefunden:[/red] {', '.join(map(str, missing))}")
        raise typer.Exit(1)

    excluded = cfg.exclusions(exclude)
    max_file_mb = max_file_mb or cfg.max_file_mb
    try:
        ocr_engine = load_ocr(ocr)
    except (OcrUnavailableError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
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
                ocr=ocr_engine,
                extensions=cfg.extensions(),
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
        hint = (
            f"auch nicht per OCR ({ocr_engine.name})"
            if ocr_engine
            else "z. B. gescannte PDFs → OCR mit `uv sync --extra ocr`"
        )
        console.print(f"[yellow]{stats.empty} Dateien ohne Text[/yellow] ({hint})")
    if stats.skipped:
        console.print(f"[dim]{stats.skipped} Wort-/Datenlisten übersprungen[/dim]")
    if stats.failed:
        console.print(f"[red]{stats.failed} Fehler[/red] – Details mit `localrag status --errors`")
    console.print(f"[dim]Index: {db}[/dim]")


def _make_watcher(db: Path, roots: list[Path], cfg: Config, exclude, embedder, lock=None):
    from localrag.ingest.scanner import Exclusions
    from localrag.ingest.watcher import FolderWatcher, WatchBatch

    def report(batch: WatchBatch) -> None:
        stamp = time.strftime("%H:%M:%S")
        if batch.error:
            console.print(f"[dim]{stamp}[/dim] [red]Fehler:[/red] {batch.error}")
            return
        s = batch.stats
        parts = [
            f"{n} {label}"
            for n, label in (
                (s.added, "neu"),
                (s.updated, "aktualisiert"),
                (s.removed, "entfernt"),
                (s.failed, "Fehler"),
                (s.embedded, "Vektoren"),
            )
            if n
        ]
        if not parts:
            return  # z. B. nur gespeichert, Inhalt unverändert
        names = ", ".join(p.name for p in batch.paths[:3]) + (" …" if len(batch.paths) > 3 else "")
        console.print(f"[dim]{stamp}[/dim] {' · '.join(parts)} [dim]({names}, {batch.seconds:.1f} s)[/dim]")

    return FolderWatcher(
        db,
        roots,
        exclusions=Exclusions(cfg.exclusions(exclude)),
        embedder=embedder,
        max_file_mb=cfg.max_file_mb,
        lock=lock,
        on_batch=report,
        extensions=cfg.extensions(),
    )


@app.command()
def watch(
    db: DbOption,
    paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Ordner. Ohne Angabe: `folders` aus der Konfiguration."),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option("--exclude", "-x", help="Zusätzlich ausschließen: Name, Muster (*backup*) oder Pfad."),
    ] = None,
    embed: Annotated[
        bool | None,
        typer.Option("--embed/--no-embed", help="Vektoren erzeugen (Standard: wenn installiert)."),
    ] = None,
    catch_up: Annotated[
        bool, typer.Option("--catch-up/--no-catch-up", help="Beim Start einmal alles abgleichen.")
    ] = True,
) -> None:
    """Beobachtet Ordner und hält den Index aktuell (neue, geänderte, gelöschte Dateien).

    Läuft der Dienst ohnehin, ist `localrag serve --watch` sparsamer: ein Prozess, ein geladenes Modell.
    """
    cfg = _config_or_exit()
    roots = paths or list(cfg.folders)
    if not roots:
        console.print(
            f"[red]Keine Ordner angegeben.[/red] `localrag watch <ordner>` oder `folders` in {config_path()}."
        )
        raise typer.Exit(2)
    missing = [p for p in roots if not p.exists()]
    if missing:
        console.print(f"[red]Nicht gefunden:[/red] {', '.join(map(str, missing))}")
        raise typer.Exit(1)
    if not db.exists() and catch_up is False:
        console.print(f"[red]Kein Index unter {db}[/red] – mit --catch-up starten oder zuerst indexieren.")
        raise typer.Exit(1)

    embedder = None
    if embed is not False:
        has_vectors = False
        if db.exists():
            with Database(db) as database:
                store = VectorStore(database)
                has_vectors = bool(store.available and store.model)
                if has_vectors:  # dasselbe Modell wie der Index
                    embedder = _load_embedder_for_search(database, "auto")
        if not has_vectors:
            embedder = _load_embedder_for_index(embed, None)

    watcher = _make_watcher(db, roots, cfg, exclude, embedder)
    model = embedder.name if embedder else "ohne Vektoren"
    console.print(f"[green]Beobachte[/green] {', '.join(map(str, watcher.roots))} · {model} · Index: {db}")
    if catch_up:
        console.print("[dim]Gleiche zuerst den Index ab …[/dim]")
    console.print("[dim]Beenden mit Strg+C[/dim]")
    watcher.start(catch_up=catch_up)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("Beende …")
    finally:
        watcher.stop()


@app.command()
def search(
    query: Annotated[
        str,
        typer.Argument(
            help="Suchanfrage in natürlicher Sprache oder Stichworte. Filter: type:pdf, in:Ordner, "
            "-in:Ordner, after:2024-01, before:2024-06-30"
        ),
    ],
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
    rerank: RerankOption = False,
) -> None:
    """Durchsucht den Index: Stichworte (BM25), Bedeutung (Vektoren) oder beides (hybrid).

    Beispiel: localrag search "Stromrechnung type:pdf in:Rechnungen after:2024"
    """
    if not db.exists():
        console.print(f"[red]Kein Index unter {db}[/red] – zuerst `localrag index <ordner>` ausführen.")
        raise typer.Exit(1)
    filter_note = _filter_note(query)
    if mode not in ("auto", "keyword", "semantic", "hybrid"):
        console.print(f"[red]Unbekannter Modus {mode!r}[/red]")
        raise typer.Exit(2)

    # Der Dienst rerankt nur, wenn er mit --rerank gestartet wurde → bei --rerank lokal arbeiten
    service = None if (local or rerank) else _service_for(db)
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
            reranker = _load_reranker(rerank)
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
                    reranker=reranker,
                )
            except EmbeddingModelMismatchError as exc:
                console.print(f"[red]{exc}[/red]")
                raise typer.Exit(1) from exc
            elapsed_ms = (time.perf_counter() - started) * 1000
            if reranker is not None:
                resolved += " + rerank"

    if as_json:
        payload = [
            {
                "path": h.path,
                "page": h.page,
                "heading": h.heading_path,
                "score": h.score,
                "snippet": h.snippet,
                "duplicates": list(h.duplicates),
            }
            for h in hits
        ]
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if not hits:
        console.print(f"Keine Treffer ({elapsed_ms:.0f} ms){filter_note}.")
        return

    for i, h in enumerate(hits, start=1):
        location = page_label(h.filetype, h.page) if h.page else (h.heading_path or "")
        header = Text.assemble((f"{i:>2}. ", "bold"), (h.path, "cyan"))
        if location:
            header.append(f"  · {location}", style="dim")
        console.print(header)
        snippet = Text("    " + h.snippet)
        snippet.highlight_words(h.matched_words, style="bold yellow", case_sensitive=False)
        console.print(snippet)
        if h.duplicates:
            names = ", ".join(Path(p).name for p in h.duplicates[:2])
            more = f" und {len(h.duplicates) - 2} weitere" if len(h.duplicates) > 2 else ""
            console.print(f"    [dim]+ gleicher Inhalt in {names}{more}[/dim]")
    console.print(f"[dim]{len(hits)} Treffer in {elapsed_ms:.0f} ms ({resolved}){filter_note}[/dim]")


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
    rerank: RerankOption = False,
) -> None:
    """Beantwortet eine Frage aus deinen Dateien – mit Quellenangaben (lokales LLM, z. B. LM Studio).

    Filter wie bei `search`, z. B. "Wie hoch war der Abschlag? in:Stadtwerke after:2024"
    """
    if not db.exists():
        console.print(f"[red]Kein Index unter {db}[/red] – zuerst `localrag index <ordner>` ausführen.")
        raise typer.Exit(1)
    _filter_note(question)
    if mode not in ("auto", "keyword", "semantic", "hybrid"):
        console.print(f"[red]Unbekannter Modus {mode!r}[/red]")
        raise typer.Exit(2)

    # Der Dienst nutzt seine eigene LLM-Konfiguration → bei explizitem --llm-url lokal arbeiten
    service = None if (local or llm_url or rerank) else _service_for(db)
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
        reranker = _load_reranker(rerank)
        resolved = "keyword" if embedder is None else ("hybrid" if mode == "auto" else mode)
        started = time.perf_counter()
        sources = retrieve_sources(
            database,
            question,
            embedder=embedder,
            mode=resolved,
            k=k,
            filetypes=filetype,  # type: ignore[arg-type]
            reranker=reranker,
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
            location = f"  · {page_label(h.filetype, h.page)}" if h.page else ""
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


def _filter_note(query: str) -> str:
    """Warnt vor unlesbaren Filtern und liefert eine kurze Angabe der aktiven Filter für die Fußzeile."""
    _, flt = parse_query(query)
    for err in flt.errors:
        console.print(f"[yellow]{err}[/yellow]")
    return f" · Filter: {flt.describe()}" if flt.active else ""


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


def _config_or_exit() -> Config:
    try:
        return load_config()
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc


AUTOSTART_NAME = "LocalRAG.vbs"
SENDTO_NAME = "LocalRAG (indexieren).vbs"


def _popup(text: str, error: bool = False) -> None:
    """Kurze Meldung als Windows-Dialog – für „Senden an“, wo es kein Konsolenfenster gibt."""
    if sys.platform != "win32":
        typer.echo(text)
        return
    import ctypes

    flags = 0x10 if error else 0x40  # MB_ICONERROR / MB_ICONINFORMATION
    ctypes.windll.user32.MessageBoxW(0, text, "LocalRAG", flags | 0x40000)  # MB_TOPMOST


def _add_summary(result: dict) -> str:
    parts = [
        f"{result[k]} {label}"
        for k, label in (
            ("added", "neu"),
            ("updated", "aktualisiert"),
            ("unchanged", "unverändert"),
            ("skipped", "übersprungen"),
            ("empty", "ohne Text"),
            ("failed", "Fehler"),
        )
        if result.get(k)
    ]
    lines = [f"{result.get('scanned', 0)} Dateien geprüft: " + (", ".join(parts) or "nichts zu tun")]
    for doc in result.get("documents", [])[:8]:
        mark = {"ok": "✓", "error": "✗", "skipped": "–", "empty": "○"}.get(doc["status"], "?")
        extra = f" ({doc['error']})" if doc.get("error") else ""
        lines.append(f"{mark} {Path(doc['path']).name}{extra}")
    if len(result.get("documents", [])) > 8:
        lines.append(f"… und {len(result['documents']) - 8} weitere")
    return "\n".join(lines)


@app.command()
def add(
    paths: Annotated[list[Path], typer.Argument(help="Dateien oder Ordner.")],
    db: DbOption,
    popup: Annotated[
        bool, typer.Option("--popup", help="Ergebnis als Windows-Dialog (für „Senden an“).")
    ] = False,
) -> None:
    """Indexiert Dateien oder Ordner sofort – auch außerhalb der beobachteten Ordner.

    Läuft der Dienst, rechnet er mit seinem geladenen Modell (schnell, kein zweites Modell im Grafikspeicher).
    """
    import os

    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        msg = "Nicht gefunden: " + ", ".join(missing)
        _popup(msg, error=True) if popup else console.print(f"[red]{msg}[/red]")
        raise typer.Exit(1)
    service = _service_for(db)
    try:
        if service is not None:
            with service:
                result = service.index([str(p.resolve()) for p in paths])
        else:
            cfg = _config_or_exit()
            embedder = None
            with Database(db) as database:
                store = VectorStore(database)
                if store.available and store.model:
                    embedder = _load_embedder_for_search(database, "auto")
                stats = index_paths(
                    database,
                    paths,
                    embedder=embedder,
                    extensions=cfg.extensions(),
                    excluded_dirs=cfg.exclusions(),
                    max_file_mb=cfg.max_file_mb,
                )
                docs = []
                for p in paths:
                    root = str(p.resolve())
                    docs += [
                        dict(r)
                        for r in database.conn.execute(
                            "SELECT id, path, status, error FROM documents "
                            "WHERE path = ? OR path LIKE ? LIMIT 50",
                            (root, root.rstrip("\\/") + os.sep + "%"),
                        )
                    ]
            result = {**{k: getattr(stats, k) for k in (
                "scanned", "added", "updated", "unchanged", "removed", "failed", "skipped", "empty"
            )}, "documents": docs}  # fmt: skip
    except (RuntimeError, EmbeddingModelMismatchError) as exc:
        _popup(str(exc), error=True) if popup else console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    summary = _add_summary(result)
    if popup:
        _popup(summary, error=bool(result.get("failed")))
    else:
        console.print(summary, markup=False, highlight=False)


def sendto_script(python: str, env: dict[str, str]) -> str:
    """VBScript fürs „Senden an“-Menü: übergibt die markierten Dateien an `localrag add --popup`."""

    def q(text: str) -> str:
        return text.replace('"', '""')

    lines = ['Set sh = CreateObject("WScript.Shell")']
    lines += [f'sh.Environment("Process")("{q(k)}") = "{q(v)}"' for k, v in sorted(env.items())]
    lines += [
        f'cmd = """{q(python)}"" -m localrag.cli add --popup"',
        "For Each a In WScript.Arguments",
        '  cmd = cmd & " """ & a & """"',
        "Next",
        "sh.Run cmd, 0, False",
    ]
    return "\r\n".join(lines) + "\r\n"


@app.command()
def sendto(
    enable: Annotated[
        bool | None,
        typer.Option("--enable/--disable", help="Eintrag im Explorer-Menü „Senden an“ anlegen/entfernen."),
    ] = None,
) -> None:
    """Rechtsklick → „Senden an“ → „LocalRAG (indexieren)“: markierte Dateien sofort in den Index."""
    import os

    if sys.platform != "win32":
        console.print("[red]Nur unter Windows.[/red]")
        raise typer.Exit(1)
    target = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "SendTo" / SENDTO_NAME
    if enable is None:
        console.print(f"„Senden an“ ist [bold]{'an' if target.exists() else 'aus'}[/bold] ({target})")
        return
    if not enable:
        target.unlink(missing_ok=True)
        console.print("[green]Eintrag entfernt.[/green]")
        return
    env = {k: v for k in ("LOCALRAG_MODELS", "LOCALRAG_CONFIG", "LOCALRAG_DB") if (v := os.environ.get(k))}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(sendto_script(sys.executable, env), encoding="utf-16")
    console.print(f"[green]Eintrag angelegt:[/green] Rechtsklick → Senden an → {target.stem}")


def autostart_script(python: str, log: Path, env: dict[str, str]) -> str:
    """VBScript für den Autostart-Ordner: startet `serve --watch` unsichtbar und schreibt ein Log."""

    def q(text: str) -> str:
        return text.replace('"', '""')

    lines = ['Set sh = CreateObject("WScript.Shell")']
    lines += [f'sh.Environment("Process")("{q(k)}") = "{q(v)}"' for k, v in sorted(env.items())]
    command = f'cmd /c ""{python}" -m localrag.cli serve --watch > "{log}" 2>&1"'
    lines.append(f'sh.Run "{q(command)}", 0, False')
    return "\r\n".join(lines) + "\r\n"


def startup_folder() -> Path:
    import os

    return Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


@app.command()
def autostart(
    enable: Annotated[
        bool | None, typer.Option("--enable/--disable", help="Beim Anmelden starten bzw. nicht mehr.")
    ] = None,
) -> None:
    """Startet `localrag serve --watch` bei jeder Windows-Anmeldung im Hintergrund (ohne Fenster).

    Ordner und Index kommen aus der Konfiguration. Log: ~/.localrag/serve.log
    """
    import os

    if sys.platform != "win32":
        console.print("[red]Nur unter Windows.[/red] Unter Linux/macOS z. B. systemd bzw. launchd nutzen.")
        raise typer.Exit(1)
    target = startup_folder() / AUTOSTART_NAME
    if enable is None:
        state = "an" if target.exists() else "aus"
        console.print(f"Autostart ist [bold]{state}[/bold] ({target})")
        return
    if not enable:
        target.unlink(missing_ok=True)
        console.print("[green]Autostart entfernt.[/green]")
        return
    cfg = _config_or_exit()
    if not cfg.folders:
        console.print(
            f"[red]Erst `folders` in {config_path()} eintragen[/red] – sonst gibt es nichts zu beobachten."
        )
        raise typer.Exit(2)
    env = {k: v for k in ("LOCALRAG_MODELS", "LOCALRAG_CONFIG", "LOCALRAG_DB") if (v := os.environ.get(k))}
    log = config_path().parent / "serve.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(autostart_script(sys.executable, log, env), encoding="utf-16")
    console.print(f"[green]Autostart eingerichtet:[/green] {target}")
    console.print(
        f"[dim]Startet ab der nächsten Anmeldung. Sofort starten: localrag serve --watch · Log: {log}[/dim]"
    )


@app.command("config")
def config_command(
    init: Annotated[
        bool, typer.Option("--init", help="Vorlage anlegen, falls noch keine existiert.")
    ] = False,
) -> None:
    """Zeigt die Konfiguration (Ordner, Ausschlüsse, Index) oder legt mit --init eine Vorlage an."""
    path = config_path()
    if init:
        if path.exists():
            console.print(f"Konfiguration existiert schon: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
            console.print(f"[green]Vorlage angelegt:[/green] {path}")
        return
    cfg = _config_or_exit()
    console.print(
        f"[bold]Konfiguration[/bold] {path}" + ("" if cfg.source else " [dim](nicht vorhanden)[/dim]")
    )
    console.print(f"Index:       {default_db_path()}")
    console.print("Ordner:      " + (", ".join(map(str, cfg.folders)) or "[dim]keine[/dim]"))
    console.print("Ausschlüsse: " + (", ".join(cfg.exclude) or "[dim]keine eigenen[/dim]"))
    if cfg.default_excludes:
        console.print(f"[dim]+ eingebaut: {', '.join(sorted(DEFAULT_EXCLUDED_DIRS))}[/dim]")
    console.print(f"Max. Dateigröße: {cfg.max_file_mb:g} MB")


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
        f"{by_status.get('empty', 0)}, übersprungen {by_status.get('skipped', 0)}, "
        f"Fehler {by_status.get('error', 0)}) · Chunks: {st['chunks']}"
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
