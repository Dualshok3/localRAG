"""Suche in drei Modi: Stichwort (BM25), semantisch (Vektoren) und hybrid (beide per RRF).

Reciprocal Rank Fusion: score(chunk) = Σ 1 / (RRF_K + rang_in_liste)
Die Scores der beiden Verfahren sind nicht vergleichbar (BM25 ist unbeschränkt, Kosinus liegt
zwischen -1 und 1). RRF nutzt deshalb nur die Ränge und braucht keine Normalisierung.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from localrag.index.db import Database
from localrag.index.embedder import Embedder
from localrag.index.text_norm import query_terms
from localrag.index.vector_store import VectorStore
from localrag.retrieval.keyword import SearchHit, keyword_ranking, make_snippet

Mode = Literal["auto", "keyword", "semantic", "hybrid"]
RRF_K = 60


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def resolve_mode(
    db: Database, mode: Mode, embedder: Embedder | None
) -> Literal["keyword", "semantic", "hybrid"]:
    if mode != "auto":
        return mode
    return "hybrid" if embedder is not None and VectorStore(db).available else "keyword"


def search(
    db: Database,
    query: str,
    *,
    mode: Mode = "auto",
    embedder: Embedder | None = None,
    limit: int = 10,
    filetypes: list[str] | None = None,
    one_per_document: bool = True,
    candidates: int = 50,
) -> list[SearchHit]:
    resolved = resolve_mode(db, mode, embedder)
    if resolved in ("semantic", "hybrid") and embedder is None:
        raise ValueError(f"Modus '{resolved}' braucht einen Embedder")

    types = [t.lower().lstrip(".") for t in filetypes] if filetypes else None
    terms = query_terms(query)
    if not terms and not query.strip():
        return []
    # Bei einem Treffer pro Dokument mehr Kandidaten holen, damit nach dem Deduplizieren genug übrig sind
    pool = max(candidates, limit * 8 if one_per_document else limit)

    keyword_ranked: list[tuple[int, float]] = []
    vector_ranked: list[tuple[int, float]] = []
    if resolved in ("keyword", "hybrid"):
        keyword_ranked = keyword_ranking(db, terms, k=pool, filetypes=types)
    if resolved in ("semantic", "hybrid"):
        assert embedder is not None
        store = VectorStore(db)
        store.ensure(embedder)  # wirft bei Modellwechsel einen verständlichen Fehler
        # Vektorsuche kennt keine Filter → mehr holen und danach filtern
        hits = store.search(embedder.embed_query(query), k=pool * (4 if types else 1))
        vector_ranked = [(h.chunk_id, h.similarity) for h in hits]

    if resolved == "keyword":
        fused = keyword_ranked
    elif resolved == "semantic":
        fused = vector_ranked
    else:
        fused = rrf([[i for i, _ in keyword_ranked], [i for i, _ in vector_ranked]])

    rows = _fetch_chunks(db, [i for i, _ in fused])
    results: list[SearchHit] = []
    seen_docs: set[int] = set()
    for chunk_id, score in fused:
        row = rows.get(chunk_id)
        if row is None or (types and row["filetype"] not in types):
            continue
        if one_per_document:
            if row["document_id"] in seen_docs:
                continue
            seen_docs.add(row["document_id"])
        snippet, matched = make_snippet(row["text"], terms)
        results.append(
            SearchHit(
                chunk_id=chunk_id,
                document_id=row["document_id"],
                path=row["path"],
                filetype=row["filetype"],
                page=row["page"],
                heading_path=row["heading_path"],
                text=row["text"],
                score=score,
                snippet=snippet,
                matched_words=matched,
            )
        )
        if len(results) >= limit:
            break
    return results


def _fetch_chunks(db: Database, ids: list[int]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for start in range(0, len(ids), 500):  # SQLite-Parameterlimit
        batch = ids[start : start + 500]
        rows = db.conn.execute(
            "SELECT c.id, c.document_id, c.text, c.page, c.heading_path, d.path, d.filetype "
            "FROM chunks c JOIN documents d ON d.id = c.document_id "
            f"WHERE c.id IN ({','.join('?' * len(batch))})",
            batch,
        )
        out.update({int(r["id"]): dict(r) for r in rows})
    return out
