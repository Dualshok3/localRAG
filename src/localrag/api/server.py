"""Lokaler Hintergrunddienst: hält das Embedding-Modell geladen, damit Suchen in Millisekunden antworten.

Start:  localrag serve            (Standard: http://127.0.0.1:8765, nur von diesem Rechner erreichbar)

Endpunkte
- GET  /health                     Status, Index-Pfad, Embedding-Modell
- GET  /search?q=…&mode=…&limit=…  Trefferliste als JSON
- POST /ask                        Antwort als NDJSON-Stream: sources → token… → done

Achtung: Der Dienst gibt Inhalte deiner Dateien heraus. Er bindet deshalb standardmäßig nur an
127.0.0.1 und sollte nie an 0.0.0.0 oder ins Netz freigegeben werden.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from localrag import __version__
from localrag.generation.answer import AnswerResult, retrieve_sources, stream_answer
from localrag.generation.llm import ChatClient, LLMError
from localrag.index.db import Database
from localrag.index.embedder import Embedder, load_embedder, semantic_available
from localrag.index.vector_store import EmbeddingModelMismatchError, VectorStore
from localrag.retrieval.hybrid import search
from localrag.retrieval.keyword import SearchHit
from localrag.retrieval.reranker import Reranker

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
ENV_SERVER = "LOCALRAG_SERVER"

ModeParam = Literal["auto", "keyword", "semantic", "hybrid"]


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
) -> FastAPI:
    db_path = Path(db_path).resolve()
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
