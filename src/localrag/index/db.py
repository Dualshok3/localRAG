"""SQLite-Datenhaltung: Manifest, Chunks und FTS5-Index in einer Datei (siehe ADR 0001)."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,
    hash        TEXT,
    mtime       REAL NOT NULL,
    size        INTEGER NOT NULL,
    filetype    TEXT NOT NULL,
    status      TEXT NOT NULL,          -- ok | empty | error
    error       TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    indexed_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY,
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord          INTEGER NOT NULL,
    text         TEXT NOT NULL,
    page         INTEGER,
    heading_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

-- rowid = chunks.id; enthält bereits gestemmten Text (siehe text_norm.py)
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    body,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


@dataclass(frozen=True)
class DocumentRow:
    id: int
    path: str
    hash: str | None
    mtime: float
    size: int
    filetype: str
    status: str
    error: str | None
    chunk_count: int


@dataclass(frozen=True)
class ChunkInput:
    text: str
    fts_body: str
    page: int | None
    heading_path: str | None


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.executescript(_SCHEMA)
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.conn:
            yield self.conn

    # --- Dokumente -----------------------------------------------------------------------------

    def get_document(self, path: str) -> DocumentRow | None:
        row = self.conn.execute(
            "SELECT id, path, hash, mtime, size, filetype, status, error, chunk_count "
            "FROM documents WHERE path = ?",
            (path,),
        ).fetchone()
        return DocumentRow(**dict(row)) if row else None

    def iter_document_paths(self) -> Iterator[str]:
        for (path,) in self.conn.execute("SELECT path FROM documents"):
            yield path

    def touch_document(self, doc_id: int, mtime: float, size: int) -> None:
        """Datei unverändert (gleicher Hash), nur Zeitstempel aktualisieren."""
        with self.transaction() as c:
            c.execute("UPDATE documents SET mtime = ?, size = ? WHERE id = ?", (mtime, size, doc_id))

    def upsert_document(
        self,
        *,
        path: str,
        hash: str | None,
        mtime: float,
        size: int,
        filetype: str,
        status: str,
        error: str | None,
        chunks: Iterable[ChunkInput],
    ) -> int:
        """Legt ein Dokument an oder ersetzt es samt aller Chunks – atomar in einer Transaktion."""
        chunk_list = list(chunks)
        with self.transaction() as c:
            existing = c.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()
            if existing:
                doc_id = int(existing["id"])
                self._delete_chunks(c, doc_id)
                c.execute(
                    "UPDATE documents SET hash=?, mtime=?, size=?, filetype=?, status=?, error=?, "
                    "chunk_count=?, indexed_at=? WHERE id=?",
                    (hash, mtime, size, filetype, status, error, len(chunk_list), time.time(), doc_id),
                )
            else:
                cur = c.execute(
                    "INSERT INTO documents(path, hash, mtime, size, filetype, status, error, chunk_count, "
                    "indexed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (path, hash, mtime, size, filetype, status, error, len(chunk_list), time.time()),
                )
                doc_id = int(cur.lastrowid or 0)
            for ord_, ch in enumerate(chunk_list):
                cur = c.execute(
                    "INSERT INTO chunks(document_id, ord, text, page, heading_path) VALUES (?, ?, ?, ?, ?)",
                    (doc_id, ord_, ch.text, ch.page, ch.heading_path),
                )
                c.execute("INSERT INTO chunks_fts(rowid, body) VALUES (?, ?)", (cur.lastrowid, ch.fts_body))
        return doc_id

    def delete_document(self, path: str) -> bool:
        with self.transaction() as c:
            row = c.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()
            if not row:
                return False
            self._delete_chunks(c, int(row["id"]))
            c.execute("DELETE FROM documents WHERE id = ?", (row["id"],))
        return True

    @staticmethod
    def _delete_chunks(c: sqlite3.Connection, doc_id: int) -> None:
        # FTS5-Zeilen hängen nicht am Fremdschlüssel → explizit löschen.
        c.execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE document_id = ?)", (doc_id,)
        )
        c.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))

    # --- Statistik -----------------------------------------------------------------------------

    def stats(self) -> dict[str, object]:
        c = self.conn
        by_status = dict(c.execute("SELECT status, COUNT(*) FROM documents GROUP BY status").fetchall())
        by_type = dict(
            c.execute("SELECT filetype, COUNT(*) FROM documents GROUP BY filetype ORDER BY 2 DESC").fetchall()
        )
        chunks = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"by_status": by_status, "by_type": by_type, "chunks": chunks}

    def errors(self, limit: int = 50) -> list[tuple[str, str]]:
        rows = self.conn.execute(
            "SELECT path, error FROM documents WHERE status = 'error' ORDER BY path LIMIT ?", (limit,)
        ).fetchall()
        return [(r["path"], r["error"]) for r in rows]
