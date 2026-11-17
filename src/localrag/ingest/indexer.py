"""Inkrementelle Indexierung: nur neue/geänderte Dateien verarbeiten, gelöschte entfernen."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from localrag.config import DEFAULT_EXCLUDED_DIRS, DEFAULT_MAX_FILE_MB
from localrag.index.db import ChunkInput, Database
from localrag.index.embedder import Embedder
from localrag.index.text_norm import normalize
from localrag.index.vector_store import VectorStore
from localrag.ingest.chunker import Chunk, chunk_sections
from localrag.ingest.ocr import OcrEngine
from localrag.ingest.parsers import ParseError, parse
from localrag.ingest.scanner import FileInfo, is_under, iter_files

ProgressCallback = Callable[[FileInfo, str], None]


@dataclass
class IndexStats:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    empty: int = 0
    failed: int = 0
    chunks: int = 0
    embedded: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


def file_hash(path: Path) -> str:
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fts_body(path: Path, chunk: Chunk) -> str:
    """Dateiname und Überschriften mitindexieren.

    So ist „Mietvertrag_2021.pdf“ auch über den Namen auffindbar.
    """
    parts = [path.stem, chunk.heading_path or "", chunk.text]
    return normalize(" ".join(parts))


OCR_TRIED = "kein Text, auch nicht per OCR"
# Erhöhen, wenn Parser Dateien lesen können, an denen sie vorher scheiterten → einmaliger Neuversuch
PARSER_VERSION = 2


def index_file(
    db: Database, info: FileInfo, file_digest: str, ocr: OcrEngine | None = None
) -> tuple[str, int, str | None]:
    """Parst und speichert eine Datei. Rückgabe: (status, Anzahl Chunks, Fehlermeldung)."""
    status, error = "ok", None
    chunks: list[Chunk] = []
    try:
        chunks = chunk_sections(parse(info.path, ocr=ocr))
        if not chunks:
            status = "empty"
            # merken, dass OCR schon versucht wurde → beim nächsten Lauf nicht erneut
            error = OCR_TRIED if ocr is not None and info.filetype == "pdf" else None
    except ParseError as exc:
        status, error = "error", str(exc)
    except Exception as exc:  # kaputte Dateien dürfen den Lauf nie abbrechen
        status, error = "error", f"{type(exc).__name__}: {exc}"

    db.upsert_document(
        path=str(info.path),
        hash=file_digest,
        mtime=info.mtime,
        size=info.size,
        filetype=info.filetype,
        status=status,
        error=error,
        chunks=[
            ChunkInput(text=c.text, fts_body=fts_body(info.path, c), page=c.page, heading_path=c.heading_path)
            for c in chunks
        ],
    )
    return status, len(chunks), error


def index_paths(
    db: Database,
    roots: Iterable[Path],
    *,
    progress: ProgressCallback | None = None,
    excluded_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
    max_file_mb: float = DEFAULT_MAX_FILE_MB,
    embedder: Embedder | None = None,
    reembed: bool = False,
    embed_progress: Callable[[int, int], None] | None = None,
    ocr: OcrEngine | None = None,
) -> IndexStats:
    """Text-Index aktualisieren und – falls ein Embedder übergeben wird – fehlende Vektoren erzeugen.

    Mit ``ocr`` werden gescannte PDFs per Texterkennung gelesen, auch solche, die früher ohne OCR als
    leer indexiert wurden.
    """
    roots = [Path(r).resolve() for r in roots]
    stats = IndexStats()
    seen: set[str] = set()
    files = list(iter_files(roots, excluded_dirs=excluded_dirs, max_file_mb=max_file_mb))
    # Ältere Indizes kennen den Eintrag noch nicht → gelten als Version 1
    has_docs = db.conn.execute("SELECT 1 FROM documents LIMIT 1").fetchone() is not None
    stored_version = db.get_meta("parser_version") or ("1" if has_docs else None)
    parsers_improved = stored_version not in (None, str(PARSER_VERSION))

    for info in files:
        stats.scanned += 1
        path = str(info.path)
        seen.add(path)
        existing = db.get_document(path)
        retry_ocr = (
            ocr is not None
            and existing is not None
            and existing.status == "empty"
            and existing.filetype == "pdf"
            and existing.error != OCR_TRIED
        ) or (parsers_improved and existing is not None and existing.status in ("error", "empty"))

        if existing and not retry_ocr and existing.mtime == info.mtime and existing.size == info.size:
            stats.unchanged += 1
            _report(progress, info, "unchanged")
            continue

        digest = file_hash(info.path)
        if existing and not retry_ocr and existing.hash == digest:
            db.touch_document(existing.id, info.mtime, info.size)
            stats.unchanged += 1
            _report(progress, info, "unchanged")
            continue

        status, n_chunks, error = index_file(db, info, digest, ocr=ocr)
        stats.chunks += n_chunks
        if status == "error":
            stats.failed += 1
            stats.errors.append((path, error or ""))
        elif status == "empty":
            stats.empty += 1
        if existing:
            stats.updated += 1
        else:
            stats.added += 1
        _report(progress, info, status)

    # Dateien, die im Index stehen, aber unter den gescannten Wurzeln nicht mehr existieren
    for path in list(db.iter_document_paths()):
        if path not in seen and any(is_under(path, r) for r in roots) and not Path(path).exists():
            db.delete_document(path)
            stats.removed += 1
    db.set_meta("parser_version", str(PARSER_VERSION))

    if embedder is not None:
        store = VectorStore(db)
        if reembed:
            store.reset()
        store.ensure(embedder)
        total = store.pending_count()
        if embed_progress:
            embed_progress(0, total)
        stats.embedded = store.embed_pending(
            embedder, progress=(lambda n: embed_progress(n, total)) if embed_progress else None
        )

    return stats


def _report(progress: ProgressCallback | None, info: FileInfo, status: str) -> None:
    if progress:
        progress(info, status)
