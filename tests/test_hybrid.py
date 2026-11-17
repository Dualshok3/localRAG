import sqlite3
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from localrag.index.db import Database
from localrag.index.embedder import HashingEmbedder
from localrag.index.vector_store import EmbeddingModelMismatchError, VectorStore
from localrag.ingest.indexer import index_paths
from localrag.retrieval.hybrid import rrf, search


class SynonymEmbedder(HashingEmbedder):
    """Simuliert „Sprachverständnis“: bildet Katze auf Haustier ab – deterministisch und ohne Modell."""

    SYNONYMS: ClassVar[dict[str, str]] = {"Katze": "Haustiere", "katze": "Haustiere"}

    def __init__(self) -> None:
        super().__init__(dim=128)
        self.name = "synonym-test"

    def _vector(self, text: str) -> np.ndarray:
        for word, replacement in self.SYNONYMS.items():
            text = text.replace(word, replacement)
        return super()._vector(text)


def names(hits) -> list[str]:
    return [Path(h.path).name for h in hits]


def test_rrf_rewards_items_ranked_by_both_lists():
    fused = rrf([[1, 2, 3], [3, 4, 1]])
    assert [i for i, _ in fused][:2] == [1, 3]
    assert {i for i, _ in fused} == {1, 2, 3, 4}


def test_index_creates_vectors_for_every_chunk(db: Database, corpus: Path):
    stats = index_paths(db, [corpus], embedder=HashingEmbedder())
    store = VectorStore(db)
    assert stats.embedded == stats.chunks > 0
    assert store.pending_count() == 0
    assert db.get_meta("embedding_model") == "hashing-256"
    # Zweiter Lauf: nichts mehr zu tun
    assert index_paths(db, [corpus], embedder=HashingEmbedder()).embedded == 0


def test_semantic_mode_finds_paraphrase_that_bm25_misses(db: Database, tmp_path: Path):
    root = tmp_path / "d"
    root.mkdir()
    (root / "miete.md").write_text("Die Haltung von Haustiere bedarf der Zustimmung.", encoding="utf-8")
    (root / "rezept.md").write_text("Lasagne mit Béchamel backen.", encoding="utf-8")
    emb = SynonymEmbedder()
    index_paths(db, [root], embedder=emb)

    assert search(db, "Katze", mode="keyword") == []
    assert names(search(db, "Katze", mode="semantic", embedder=emb))[0] == "miete.md"
    assert names(search(db, "Katze", mode="hybrid", embedder=emb))[0] == "miete.md"


def test_hybrid_keeps_exact_keyword_hits(db: Database, corpus: Path):
    emb = HashingEmbedder()
    index_paths(db, [corpus], embedder=emb)
    assert names(search(db, "Urlaubstage", mode="hybrid", embedder=emb))[0] == "Arbeitsvertrag.docx"


def test_auto_mode_falls_back_to_keyword_without_vectors(db: Database, corpus: Path):
    index_paths(db, [corpus])
    hits = search(db, "Kündigungsfrist", mode="auto", embedder=HashingEmbedder())
    assert names(hits)[0] == "Mietvertrag_2021.pdf"


def test_filetype_filter_applies_to_vector_hits(db: Database, corpus: Path):
    emb = HashingEmbedder()
    index_paths(db, [corpus], embedder=emb)
    hits = search(db, "Kündigungsfrist Urlaub", mode="semantic", embedder=emb, filetypes=["docx"])
    assert hits and {h.filetype for h in hits} == {"docx"}


def test_deleted_documents_lose_their_vectors(db: Database, corpus: Path):
    emb = HashingEmbedder()
    index_paths(db, [corpus], embedder=emb)
    (corpus / "notizen.md").unlink()
    index_paths(db, [corpus], embedder=emb)
    vec_rows = db.conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
    chunk_rows = db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert vec_rows == chunk_rows


def test_model_switch_requires_reembed(db: Database, corpus: Path):
    index_paths(db, [corpus], embedder=HashingEmbedder(256))
    with pytest.raises(EmbeddingModelMismatchError):
        index_paths(db, [corpus], embedder=HashingEmbedder(64))
    stats = index_paths(db, [corpus], embedder=HashingEmbedder(64), reembed=True)
    assert stats.embedded > 0
    assert db.get_meta("embedding_model") == "hashing-64"


def test_semantic_mode_without_embedder_is_an_error(db: Database, corpus: Path):
    index_paths(db, [corpus])
    with pytest.raises(ValueError):
        search(db, "x", mode="semantic")


def test_v1_database_is_migrated(tmp_path: Path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta VALUES ('schema_version', '1');
        CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, hash TEXT,
            mtime REAL NOT NULL, size INTEGER NOT NULL, filetype TEXT NOT NULL, status TEXT NOT NULL,
            error TEXT, chunk_count INTEGER NOT NULL DEFAULT 0, indexed_at REAL NOT NULL);
        CREATE TABLE chunks (id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL, ord INTEGER NOT NULL,
            text TEXT NOT NULL, page INTEGER, heading_path TEXT);
        INSERT INTO documents VALUES (1, 'x.md', 'h', 0, 1, 'md', 'ok', NULL, 1, 0);
        INSERT INTO chunks VALUES (1, 1, 0, 'alter Text', NULL, NULL);
        """
    )
    conn.commit()
    conn.close()

    with Database(path) as db:
        assert db.get_meta("schema_version") == "2"
        assert VectorStore(db).pending_count() == 1
