"""Vektorspeicher auf Basis von sqlite-vec (siehe ADR 0001).

Alle Vektorzugriffe laufen über dieses Modul, damit das Backend später austauschbar bleibt.
Vektoren sind L2-normalisiert; dann ist die Rangfolge nach euklidischem Abstand identisch mit der
nach Kosinus-Ähnlichkeit (cos = 1 - d²/2).
"""

from __future__ import annotations

from collections.abc import Callable
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
            done += len(rows)
            if progress:
                progress(len(rows))

    def search(self, query_vector: np.ndarray, k: int = 50) -> list[VectorHit]:
        if not self.available:
            return []
        rows = self.db.conn.execute(
            f"SELECT rowid, distance FROM {self.TABLE} WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (_blob(query_vector), k),
        ).fetchall()
        return [VectorHit(int(r[0]), float(r[1])) for r in rows]


def _blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()
