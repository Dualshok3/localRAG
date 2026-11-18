"""Lokaler Hintergrunddienst: hält das Embedding-Modell geladen, damit Suchen in Millisekunden antworten.

Start:  localrag serve            (Standard: http://127.0.0.1:8765, nur von diesem Rechner erreichbar)

Endpunkte
- GET  /                           Web-Oberfläche (Suche, Antwort, Quellen-Karten)
- GET  /health                     Status, Index-Pfad, Embedding-Modell
- GET  /search?q=…&mode=…&limit=…  Trefferliste als JSON
- POST /ask                        Antwort als NDJSON-Stream: sources → token… → done
- GET  /file/{document_id}         indexierte Datei ausliefern (PDF im Browser an Seite #page=N)
- POST /open                       indexierte Datei mit dem Standardprogramm öffnen bzw. im Explorer zeigen
- POST /index                      Dateien/Ordner sofort indexieren (für „Senden an“ und `localrag add`)
- POST /upload                     Datei in den Eingangsordner legen und sofort indexieren (Drag & Drop)

Achtung: Der Dienst gibt Inhalte deiner Dateien heraus. Er bindet deshalb standardmäßig nur an
127.0.0.1 und sollte nie an 0.0.0.0 oder ins Netz freigegeben werden.

Schutz gegen fremde Webseiten im selben Browser: Anfragen mit fremdem Host-Header werden abgelehnt
(DNS-Rebinding), und /open verlangt einen eigenen Header, den eine fremde Seite nicht ohne CORS-Freigabe
senden kann – die es nicht gibt. Herausgegeben werden nur Dateien, die im Index stehen.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from localrag import __version__
from localrag.generation.answer import AnswerResult, retrieve_sources, stream_answer
from localrag.generation.llm import ChatClient, LLMError
from localrag.index.db import Database
from localrag.index.embedder import Embedder, load_embedder, semantic_available
from localrag.index.vector_store import EmbeddingModelMismatchError, VectorStore
from localrag.ingest.indexer import index_paths
from localrag.ingest.parsers import SUPPORTED_EXTENSIONS
from localrag.retrieval.hybrid import search
from localrag.retrieval.keyword import SearchHit
from localrag.retrieval.reranker import Reranker

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
ENV_SERVER = "LOCALRAG_SERVER"

ModeParam = Literal["auto", "keyword", "semantic", "hybrid"]
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
UI_HEADER = "X-LocalRAG"


class IndexRequest(BaseModel):
    paths: list[str] = Field(min_length=1, max_length=500)


def safe_filename(name: str) -> str:
    """Nur der Dateiname ohne Pfadanteile und Steuerzeichen – ein Upload darf nirgendwo sonst landen."""
    from urllib.parse import unquote

    base = os.path.basename(unquote(name).replace("\\", "/")).strip()
    base = "".join(ch for ch in base if ch.isprintable() and ch not in '<>:"/\\|?*')
    if not base or base.startswith(".") or base in ("..",):
        raise HTTPException(400, "Ungültiger Dateiname")
    return base[:200]


def unique_target(folder: Path, name: str, data: bytes) -> Path:
    """Gleicher Name + gleicher Inhalt → dieselbe Datei; gleicher Name, anderer Inhalt → „Name (2).pdf“."""
    target = folder / name
    stem, suffix = os.path.splitext(name)
    n = 2
    while target.exists():
        if target.read_bytes() == data:
            return target
        target = folder / f"{stem} ({n}){suffix}"
        n += 1
    return target


class OpenRequest(BaseModel):
    document_id: int
    reveal: bool = False  # True = Ordner im Explorer zeigen statt Datei öffnen


def open_with_system(path: Path, *, reveal: bool = False) -> None:  # pragma: no cover - startet Programme
    if sys.platform == "win32":
        if reveal:
            subprocess.Popen(["explorer", "/select,", str(path)])
        else:
            os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)] if reveal else ["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path.parent if reveal else path)])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=6, ge=1, le=20)
    mode: ModeParam = "auto"
    types: list[str] | None = None
    llm_model: str | None = None


def hit_to_dict(hit: SearchHit) -> dict[str, Any]:
    d = dataclasses.asdict(hit)
    d["matched_words"] = list(hit.matched_words)
    return d


def hit_from_dict(d: dict[str, Any]) -> SearchHit:
    return SearchHit(
        **{
            **d,
            "matched_words": tuple(d.get("matched_words", ())),
            "duplicates": tuple(d.get("duplicates", ())),
        }
    )


def default_embedder_for(db_path: Path) -> Embedder | None:
    """Lädt das Modell, mit dem der Index erstellt wurde – oder keins, wenn es keine Vektoren gibt."""
    with Database(db_path) as db:
        store = VectorStore(db)
        model = store.model if store.available else None
    if not model:
        return None
    if not model.startswith("hashing-") and not semantic_available():
        return None
    return load_embedder(model)


def create_app(
    db_path: Path,
    *,
    embedder: Embedder | None = None,
    load_model: bool = True,
    llm_factory: Callable[[str | None], ChatClient] | None = None,
    reranker: Reranker | None = None,
    allowed_hosts: Iterable[str] = LOCAL_HOSTS,
    opener: Callable[..., None] = open_with_system,
    inbox: Path | None = None,
    extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
    exclusions: Iterable[str] | None = None,
    max_file_mb: float = 50,
) -> FastAPI:
    db_path = Path(db_path).resolve()
    hosts = {h.strip("[]").lower() for h in allowed_hosts}
    if embedder is None and load_model:
        embedder = default_embedder_for(db_path)
    make_llm = llm_factory or (lambda model: ChatClient(model=model))
    # Ein Modell auf der GPU, viele Anfragen in Threads → Zugriff serialisieren
    model_lock = threading.Lock()

    # Eine Verbindung je Worker-Thread statt einer pro Anfrage: Öffnen lädt sqlite-vec und prüft das
    # Schema (ein kurzer Schreibvorgang) – das kostete bei jeder Suche spürbar Zeit.
    local = threading.local()

    def connection() -> Database:
        db: Database | None = getattr(local, "db", None)
        if db is None:
            db = local.db = Database(db_path)
        return db

    app = FastAPI(title="LocalRAG", version=__version__)
    app.state.embedder = embedder
    app.state.reranker = reranker
    app.state.model_lock = model_lock  # für den Watcher in `serve --watch`
    app.state.watcher = None

    @app.middleware("http")
    async def only_local_hosts(request: Request, call_next):
        # DNS-Rebinding: eine fremde Domain, die auf 127.0.0.1 zeigt, hätte sonst Zugriff wie die eigene Seite
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]").lower()
        if hosts and host not in hosts:
            return JSONResponse({"detail": "Host nicht erlaubt"}, status_code=403)
        return await call_next(request)

    def document(document_id: int) -> Path:
        row = connection().conn.execute("SELECT path FROM documents WHERE id = ?", (document_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Datei nicht im Index")
        path = Path(row["path"])
        if not path.is_file():
            raise HTTPException(410, "Datei existiert nicht mehr")
        return path

    @app.get("/", response_class=HTMLResponse)
    def ui() -> HTMLResponse:
        html = resources.files("localrag.api").joinpath("static/index.html").read_text(encoding="utf-8")
        return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

    @app.get("/file/{document_id}")
    def file_endpoint(document_id: int) -> FileResponse:
        path = document(document_id)
        # inline: PDFs, Bilder und Text zeigt der Browser direkt an, Office-Dateien lädt er herunter
        return FileResponse(path, filename=path.name, content_disposition_type="inline")

    @app.post("/open")
    def open_endpoint(
        req: OpenRequest, x_localrag: Annotated[str | None, Header(alias=UI_HEADER)] = None
    ) -> dict[str, Any]:
        if x_localrag != "1":
            raise HTTPException(403, f"Header {UI_HEADER} fehlt")
        path = document(req.document_id)
        opener(path, reveal=req.reveal)
        return {"opened": str(path), "reveal": req.reveal}

    exts = frozenset(e.lower() for e in extensions)

    def require_ui_header(value: str | None) -> None:
        # Fremde Webseiten können diesen Header ohne CORS-Freigabe nicht senden (und die gibt es nicht)
        if value != "1":
            raise HTTPException(403, f"Header {UI_HEADER} fehlt")

    def index_now(paths: list[Path]) -> dict[str, Any]:
        """Sofort indexieren – mit dem geladenen Modell, gesperrt gegen gleichzeitige Suche und Watcher."""
        started = time.perf_counter()
        kwargs: dict[str, Any] = {"extensions": exts, "max_file_mb": max_file_mb}
        if exclusions is not None:
            kwargs["excluded_dirs"] = list(exclusions)
        with model_lock:
            db = connection()
            stats = index_paths(db, paths, embedder=embedder, **kwargs)
            docs = []
            for p in paths:
                prefix = str(p.resolve())
                rows = db.conn.execute(
                    "SELECT id, path, status, error, chunk_count FROM documents "
                    "WHERE path = ? OR path LIKE ? ESCAPE '!' ORDER BY path LIMIT 50",
                    (
                        prefix,
                        prefix.replace("!", "!!").replace("%", "!%").replace("_", "!_").rstrip("\\/")
                        + os.sep
                        + "%",
                    ),
                ).fetchall()
                docs += [dict(r) for r in rows]
        return {
            "seconds": round(time.perf_counter() - started, 2),
            "scanned": stats.scanned,
            "added": stats.added,
            "updated": stats.updated,
            "unchanged": stats.unchanged,
            "removed": stats.removed,
            "failed": stats.failed,
            "skipped": stats.skipped,
            "empty": stats.empty,
            "chunks": stats.chunks,
            "documents": docs[:50],
        }

    @app.post("/index")
    def index_endpoint(
        req: IndexRequest, x_localrag: Annotated[str | None, Header(alias=UI_HEADER)] = None
    ) -> dict[str, Any]:
        require_ui_header(x_localrag)
        paths = [Path(p) for p in req.paths]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise HTTPException(404, "Nicht gefunden: " + ", ".join(missing[:5]))
        return index_now(paths)

    @app.post("/upload")
    async def upload_endpoint(
        request: Request,
        x_localrag: Annotated[str | None, Header(alias=UI_HEADER)] = None,
        x_filename: Annotated[str | None, Header(alias="X-Filename")] = None,
    ) -> dict[str, Any]:
        require_ui_header(x_localrag)
        if inbox is None:
            raise HTTPException(503, "Kein Eingangsordner konfiguriert")
        name = safe_filename(x_filename or "")
        if os.path.splitext(name)[1].lower() not in exts:
            raise HTTPException(
                415, f"Dateityp {os.path.splitext(name)[1] or '(ohne Endung)'} wird nicht unterstützt"
            )
        limit = int(max_file_mb * 1024 * 1024)
        if int(request.headers.get("content-length") or 0) > limit:
            raise HTTPException(413, f"Datei größer als {max_file_mb:g} MB")
        data = await request.body()
        if not data:
            raise HTTPException(400, "Leere Datei")
        if len(data) > limit:
            raise HTTPException(413, f"Datei größer als {max_file_mb:g} MB")
        inbox.mkdir(parents=True, exist_ok=True)
        target = unique_target(inbox, name, data)
        if not target.exists():
            target.write_bytes(data)
        result = await run_in_threadpool(index_now, [target])
        return {"saved": str(target), **result}

    @app.get("/stats")
    def stats_endpoint() -> dict[str, Any]:
        st = connection().stats()
        return {"documents": sum(st["by_status"].values()), "chunks": st["chunks"], **st}  # type: ignore[arg-type]

    app.state.db_path = db_path

    def resolved_mode(mode: str) -> str:
        if mode == "auto":
            return "hybrid" if embedder is not None else "keyword"
        if mode != "keyword" and embedder is None:
            raise HTTPException(400, f"Modus '{mode}' braucht Vektoren – Dienst läuft ohne Embedding-Modell")
        return mode

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "db": str(db_path),
            "db_exists": db_path.exists(),
            "embedding_model": embedder.name if embedder else None,
            "reranker": reranker.name if reranker else None,
            "watching": [str(r) for r in app.state.watcher.roots] if app.state.watcher else None,
            "inbox": str(inbox) if inbox else None,
            "extensions": sorted(exts),
        }

    @app.get("/search")
    def search_endpoint(
        q: Annotated[str, Query(min_length=1, max_length=2000)],
        mode: ModeParam = "auto",
        limit: Annotated[int, Query(ge=1, le=100)] = 10,
        type: Annotated[list[str] | None, Query()] = None,
        chunks: bool = False,
    ) -> dict[str, Any]:
        resolved = resolved_mode(mode)
        started = time.perf_counter()
        try:
            with model_lock:
                hits = search(
                    connection(),
                    q,
                    mode=resolved,  # type: ignore[arg-type]
                    embedder=embedder,
                    limit=limit,
                    filetypes=type,
                    one_per_document=not chunks,
                    reranker=reranker,
                )
        except EmbeddingModelMismatchError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {
            "mode": resolved,
            "took_ms": (time.perf_counter() - started) * 1000,
            "hits": [hit_to_dict(h) for h in hits],
        }

    @app.post("/ask")
    def ask_endpoint(req: AskRequest) -> StreamingResponse:
        resolved = resolved_mode(req.mode)
        started = time.perf_counter()
        with model_lock:
            sources = retrieve_sources(
                connection(),
                req.question,
                embedder=embedder,
                mode=resolved,  # type: ignore[arg-type]
                k=req.k,
                filetypes=req.types,
                reranker=reranker,
            )
        search_ms = (time.perf_counter() - started) * 1000

        def events() -> Iterator[str]:
            result = AnswerResult(question=req.question, sources=sources)
            yield _event(
                type="sources",
                mode=resolved,
                search_ms=search_ms,
                sources=[hit_to_dict(h) for h in sources],
            )
            client = make_llm(req.llm_model)
            try:
                yield _event(type="model", model=client.model if sources else None)
                for piece in stream_answer(client, result):
                    yield _event(type="token", text=piece)
                yield _event(type="done", cited=result.cited, no_answer=result.is_no_answer)
            except LLMError as exc:
                yield _event(type="error", message=str(exc))
            finally:
                client.close()

        return StreamingResponse(events(), media_type="application/x-ndjson")

    return app


def _event(**payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def server_url() -> str:
    return os.environ.get(ENV_SERVER, f"http://{DEFAULT_HOST}:{DEFAULT_PORT}").rstrip("/")
