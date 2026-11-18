"""Vektorspeicher auf Basis von sqlite-vec (siehe ADR 0001).

Alle Vektorzugriffe laufen über dieses Modul, damit das Backend später austauschbar bleibt.
Vektoren sind L2-normalisiert; dann ist die Rangfolge nach euklidischem Abstand identisch mit der
nach Kosinus-Ähnlichkeit (cos = 1 - d²/2).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import PureWindowsPath

import numpy as np

from localrag.index.db import Database
from localrag.index.embedder import Embedder


class EmbeddingModelMismatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class VectorHit:
    chunk_id: int
    distance: float

    @property
    def similarity(self) -> float:
        return 1.0 - (self.distance**2) / 2.0


def embedding_text(path_stem: str, heading_path: str | None, text: str) -> str:
    """Kontext mit einbetten: Dateiname + Überschriften helfen dem Modell bei kurzen Abschnitten."""
    header = path_stem.replace("_", " ")
    if heading_path:
        header += f" – {heading_path}"
    return f"{header}\n\n{text}"


# --- Optionale Kopie der Vektoren im Arbeitsspeicher -------------------------------------------
# sqlite-vec vergleicht bei jeder Suche alle Vektoren direkt aus der Datei: bei 70.000 Abschnitten
# rund 300 ms. Ein lang laufender Prozess (Dienst, MCP-Server) hält sie stattdessen als Matrix im
# Speicher – eine Suche ist dann eine Matrixmultiplikation. Für einmalige CLI-Aufrufe lohnt das
# Laden nicht, deshalb nur auf Wunsch (enable_memory_cache).

_memory_enabled = False
_memory: dict[str, _Matrix] = {}
_memory_lock = threading.Lock()


@dataclass(frozen=True)
class _Matrix:
    version: str
    ids: np.ndarray  # int64
    vectors: np.ndarray  # float32, normalisiert, eine Zeile pro Abschnitt


def enable_memory_cache(enabled: bool = True) -> None:
    global _memory_enabled
    _memory_enabled = enabled
    if not enabled:
        _memory.clear()


class VectorStore:
    TABLE = "chunks_vec"

    def __init__(self, db: Database) -> None:
        self.db = db

    @property
    def model(self) -> str | None:
        return self.db.get_meta("embedding_model")

    @property
    def available(self) -> bool:
        return self.db.has_table(self.TABLE)

    def ensure(self, embedder: Embedder) -> None:
        """Legt die Vektortabelle an oder prüft, dass das gleiche Modell verwendet wird."""
        current = self.model
        if current and current != embedder.name:
            raise EmbeddingModelMismatchError(
                f"Index wurde mit '{current}' erstellt, aktuell ist '{embedder.name}'. "
                "Neu einbetten mit: localrag index <ordner> --reembed"
            )
        if not self.available:
            with self.db.transaction() as c:
                c.execute(f"CREATE VIRTUAL TABLE {self.TABLE} USING vec0(embedding float[{embedder.dim}])")
                self.db.set_meta("embedding_model", embedder.name)
                self.db.set_meta("embedding_dim", str(embedder.dim))

    def reset(self) -> None:
        with self.db.transaction() as c:
            c.execute(f"DROP TABLE IF EXISTS {self.TABLE}")
            c.execute("UPDATE chunks SET embedded = 0")
            self.db.bump_vectors_version()
            self.db.set_meta("embedding_model", None)
            self.db.set_meta("embedding_dim", None)

    def pending_count(self) -> int:
        return int(self.db.conn.execute("SELECT COUNT(*) FROM chunks WHERE embedded = 0").fetchone()[0])

    def embed_pending(
        self,
        embedder: Embedder,
        *,
        batch_size: int = 256,
        progress: Callable[[int], None] | None = None,
    ) -> int:
        """Bettet alle Chunks ohne Vektor ein. Rückgabe: Anzahl neu eingebetteter Chunks."""
        self.ensure(embedder)
        done = 0
        while True:
            rows = self.db.conn.execute(
                "SELECT c.id, c.text, c.heading_path, d.path FROM chunks c "
                "JOIN documents d ON d.id = c.document_id WHERE c.embedded = 0 ORDER BY c.id LIMIT ?",
                (batch_size,),
            ).fetchall()
            if not rows:
                return done
            texts = [
                embedding_text(PureWindowsPath(r["path"]).stem, r["heading_path"], r["text"]) for r in rows
            ]
            vectors = embedder.embed_documents(texts)
            with self.db.transaction() as c:
                c.executemany(
                    f"INSERT OR REPLACE INTO {self.TABLE}(rowid, embedding) VALUES (?, ?)",
                    [(r["id"], _blob(v)) for r, v in zip(rows, vectors, strict=True)],
                )
                c.executemany("UPDATE chunks SET embedded = 1 WHERE id = ?", [(r["id"],) for r in rows])
                self.db.bump_vectors_version()
            done += len(rows)
            if progress:
                progress(len(rows))

    def search(
        self, query_vector: np.ndarray, k: int = 50, *, candidates: Sequence[int] | None = None
    ) -> list[VectorHit]:
        """Nächste Nachbarn; mit ``candidates`` exakt nur unter diesen Abschnitten (für Filter)."""
        if not self.available:
            return []
        if candidates is not None:
            return self._search_candidates(query_vector, k, candidates)
        if _memory_enabled and str(self.db.path) != ":memory:":
            return self._search_in_memory(query_vector, k)
        rows = self.db.conn.execute(
            f"SELECT rowid, distance FROM {self.TABLE} WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (_blob(query_vector), k),
        ).fetchall()
        return [VectorHit(int(r[0]), float(r[1])) for r in rows]

    def _matrix(self) -> _Matrix:
        key = str(self.db.path.resolve())
        version = self.db.get_meta("vectors_version") or "0"
        with _memory_lock:
            cached = _memory.get(key)
            if cached is not None and cached.version == version:
                return cached
            dim = int(self.db.get_meta("embedding_dim") or 0)
            loaded = self._load_from_blocks(dim) or self._load_row_by_row(dim)
            matrix = _Matrix(version, *loaded)
            _memory[key] = matrix
            return matrix

    def _load_row_by_row(self, dim: int) -> tuple[np.ndarray, np.ndarray]:
        """Sicherer Weg über die öffentliche Schnittstelle – aber langsam (~1 ms pro Vektor)."""
        rows = self.db.conn.execute(f"SELECT rowid, embedding FROM {self.TABLE}").fetchall()
        ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        vectors = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), dim)
        return ids, vectors

    def _load_from_blocks(self, dim: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Schneller Weg: die Blöcke lesen, in denen sqlite-vec die Vektoren ablegt (je bis zu 1024 Stück).

        Das ist ein internes Speicherformat von sqlite-vec (geprüft mit v0.1.x). Passt etwas nicht – andere
        Tabellen, andere Größen, abweichende Anzahl –, gibt es None und es wird zeilenweise geladen.
        """
        t = self.TABLE
        try:
            blocks = self.db.conn.execute(
                f"SELECT c.size, c.validity, c.rowids, v.vectors FROM {t}_chunks c "
                f"JOIN {t}_vector_chunks00 v ON v.rowid = c.chunk_id ORDER BY c.chunk_id"
            ).fetchall()
            expected = int(self.db.conn.execute(f"SELECT COUNT(*) FROM {t}_rowids").fetchone()[0])
        except Exception:
            return None
        id_parts: list[np.ndarray] = []
        vec_parts: list[np.ndarray] = []
        for size, validity, rowids, vectors in blocks:
            if len(rowids) != size * 8 or len(vectors) != size * dim * 4 or len(validity) * 8 < size:
                return None
            bits = np.unpackbits(np.frombuffer(validity, dtype=np.uint8), bitorder="little")
            valid = bits[:size].astype(bool)
            id_parts.append(np.frombuffer(rowids, dtype=np.int64)[valid])
            vec_parts.append(np.frombuffer(vectors, dtype=np.float32).reshape(size, dim)[valid])
        ids = np.concatenate(id_parts) if id_parts else np.empty(0, dtype=np.int64)
        if len(ids) != expected:
            return None
        vectors = np.concatenate(vec_parts) if vec_parts else np.empty((0, dim), dtype=np.float32)
        return ids, vectors

    def _search_in_memory(self, query_vector: np.ndarray, k: int) -> list[VectorHit]:
        m = self._matrix()
        return _top_k(m.ids, m.vectors, query_vector, k)

    def _search_candidates(
        self, query_vector: np.ndarray, k: int, candidates: Sequence[int]
    ) -> list[VectorHit]:
        """Exakte Suche in einer Teilmenge. Filter wie ``in:Rechnungen`` träfen sonst oft nichts, weil
        die passenden Abschnitte nicht unter den global nächsten Nachbarn sind."""
        wanted = np.fromiter(candidates, dtype=np.int64)
        if len(wanted) == 0:
            return []
        if _memory_enabled and str(self.db.path) != ":memory:":
            m = self._matrix()
            mask = np.isin(m.ids, wanted)
            return _top_k(m.ids[mask], m.vectors[mask], query_vector, k)
        dim = int(self.db.get_meta("embedding_dim") or 0)
        ids: list[int] = []
        blobs: list[bytes] = []
        for start in range(0, len(wanted), 500):  # SQLite-Parameterlimit
            batch = [int(i) for i in wanted[start : start + 500]]
            marks = ",".join("?" * len(batch))
            for rowid, blob in self.db.conn.execute(
                f"SELECT rowid, embedding FROM {self.TABLE} WHERE rowid IN ({marks})", batch
            ):
                ids.append(int(rowid))
                blobs.append(blob)
        if not ids:
            return []
        vectors = np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(ids), dim)
        return _top_k(np.asarray(ids, dtype=np.int64), vectors, query_vector, k)


def _top_k(ids: np.ndarray, vectors: np.ndarray, query_vector: np.ndarray, k: int) -> list[VectorHit]:
    if len(ids) == 0:
        return []
    sims = vectors @ np.asarray(query_vector, dtype=np.float32)
    k = min(k, len(sims))
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    # gleiche Distanz wie sqlite-vec (euklidisch auf normalisierten Vektoren)
    dist = np.sqrt(np.clip(2.0 - 2.0 * sims[top], 0.0, None))
    return [VectorHit(int(ids[i]), float(d)) for i, d in zip(top, dist, strict=True)]


def _blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()
