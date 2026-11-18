"""Suche in drei Modi: Stichwort (BM25), semantisch (Vektoren) und hybrid (beide per RRF).

Reciprocal Rank Fusion: score(chunk) = Σ gewicht / (RRF_K + rang_in_liste)
Die Scores der beiden Verfahren sind nicht vergleichbar (BM25 ist unbeschränkt, Kosinus liegt
zwischen -1 und 1). RRF nutzt deshalb nur die Ränge und braucht keine Normalisierung.

Gewichtung: Bei Kennungen und kurzen Stichworten („WTEST-998877“, „Router Firmware“) zählt BM25 voll.
Bei ganzen Sätzen liefert BM25 oft Zufallstreffer über Allerweltswörter („Wann“, „meinem“, „Auto“), die
den richtigen Vektortreffer verdrängen – dort zählt die Stichwortliste deshalb weniger.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from localrag.index.db import Database
from localrag.index.embedder import Embedder
from localrag.index.text_norm import STOPWORDS, query_terms
from localrag.index.vector_store import VectorStore
from localrag.retrieval.filters import SearchFilter, parse_query
from localrag.retrieval.keyword import SearchHit, keyword_ranking, make_snippet
from localrag.retrieval.reranker import Reranker

Mode = Literal["auto", "keyword", "semantic", "hybrid"]
RRF_K = 60
RERANK_DEPTH = 40  # so viele Abschnitte der ersten Stufe liest der Reranker
# Reranker-Reihenfolge übernehmen. Mischen mit der ersten Stufe (z. B. 1.0) war auf 78 Fragen im kleinen
# Index gleichauf, im Index über ganze Laufwerke schlechter (Recall@1 0.69 statt 0.73).
RERANK_BLEND: float | None = None
SENTENCE_KEYWORD_WEIGHT = 0.3  # Gewicht der BM25-Liste bei Fragen in ganzen Sätzen (gemessen: 0.3 > 0.5)
SHORT_QUERY_WORDS = 3
# Bis zu so vielen Abschnitten wird bei aktivem Filter exakt in der Teilmenge gesucht, darüber global
# mit mehr Kandidaten und anschließendem Filtern.
EXACT_FILTER_LIMIT = 30_000
_CODE_LIKE = re.compile(r"\d|[A-ZÄÖÜ]{2,}|[_.]\w")  # Ziffern, Abkürzungen, Dateinamen/Kennungen
_WORD_CHARS = re.compile(r"[^\W_]+")


def keyword_weight_for(query: str) -> float:
    """Gewicht der Stichwortliste: voll bei Stichworten und Kennungen, reduziert bei ganzen Sätzen.

    Ein Satz erkennt sich an Füllwörtern („ich“, „der“, „wann“) oder einem Fragezeichen – nicht an der
    Länge: „Steuererklärung Belege Frist Finanzamt Formular“ sind fünf Stichworte, kein Satz.
    """
    words = query.split()
    if len(words) <= SHORT_QUERY_WORDS or any(_CODE_LIKE.search(w) for w in words):
        return 1.0
    has_filler = any(w.casefold() in STOPWORDS for w in _WORD_CHARS.findall(query))
    return SENTENCE_KEYWORD_WEIGHT if has_filler or "?" in query else 1.0


def rrf(
    rankings: Sequence[Sequence[int]], k: int = RRF_K, weights: Sequence[float] | None = None
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for n, ranking in enumerate(rankings):
        weight = weights[n] if weights else 1.0
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
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
    collapse_duplicates: bool = True,
    reranker: Reranker | None = None,
    rerank_depth: int = RERANK_DEPTH,
    keyword_weight: float | None = None,
    rerank_blend: float | None = RERANK_BLEND,
    flt: SearchFilter | None = None,
    parse_filters: bool = True,
) -> list[SearchHit]:
    """Trefferliste, ein Treffer pro Datei (außer ``one_per_document=False``).

    keyword_weight: Gewicht der BM25-Liste im Hybridmodus; None = automatisch nach Art der Anfrage.
    rerank_blend: None = Reihenfolge ganz dem Reranker überlassen; sonst sein Gewicht beim Mischen mit
    der Reihenfolge der ersten Stufe (1.0 = beide gleich).
    parse_filters: Angaben wie ``type:pdf in:Rechnungen after:2024`` aus der Anfrage lesen (filters.py).
    """
    resolved = resolve_mode(db, mode, embedder)
    if resolved in ("semantic", "hybrid") and embedder is None:
        raise ValueError(f"Modus '{resolved}' braucht einen Embedder")

    flt = flt or SearchFilter()
    if parse_filters:
        query, parsed = parse_query(query)
        flt = merge_filters(flt, parsed)
    flt = flt.with_types(filetypes)
    terms = query_terms(query)
    if not terms and not query.strip():
        return []
    # Bei einem Treffer pro Dokument mehr Kandidaten holen, damit nach dem Deduplizieren genug übrig sind
    pool = max(candidates, limit * 8 if one_per_document else limit)

    keyword_ranked: list[tuple[int, float]] = []
    vector_ranked: list[tuple[int, float]] = []
    if resolved in ("keyword", "hybrid"):
        keyword_ranked = keyword_ranking(db, terms, k=pool, flt=flt)
    if resolved in ("semantic", "hybrid") and query.strip():
        assert embedder is not None
        store = VectorStore(db)
        store.ensure(embedder)  # wirft bei Modellwechsel einen verständlichen Fehler
        candidates = _filtered_chunk_ids(db, flt) if flt.active else None
        if candidates is not None and len(candidates) > EXACT_FILTER_LIMIT:
            # großer Teil des Index → global suchen, mehr holen und danach filtern
            hits = store.search(embedder.embed_query(query), k=pool * 4)
        else:
            hits = store.search(embedder.embed_query(query), k=pool, candidates=candidates)
        vector_ranked = [(h.chunk_id, h.similarity) for h in hits]

    if resolved == "keyword":
        fused = keyword_ranked
    elif resolved == "semantic":
        fused = vector_ranked
    else:
        weight = keyword_weight_for(query) if keyword_weight is None else keyword_weight
        fused = rrf([[i for i, _ in keyword_ranked], [i for i, _ in vector_ranked]], weights=[weight, 1.0])

    rows = _fetch_chunks(db, [i for i, _ in fused])
    if flt.active:
        fused = [
            (i, s)
            for i, s in fused
            if i in rows and flt.matches(rows[i]["path"], rows[i]["filetype"], rows[i]["mtime"])
        ]
    if reranker is not None and fused:
        fused = _rerank(reranker, query, fused, rows, rerank_depth, rerank_blend)
    kept: list[tuple[int, float, dict, frozenset[str], list[str]]] = []
    seen_docs: set[int] = set()
    for chunk_id, score in fused:
        row = rows.get(chunk_id)
        if row is None:
            continue
        if one_per_document and row["document_id"] in seen_docs:
            continue
        words = _word_set(row["text"])
        if collapse_duplicates:
            original = next((k for k in kept if _near_duplicate(words, k[3])), None)
            if original is not None:
                # Gleicher Inhalt in anderer Datei (Backup, Kopie, PDF-Export) → beim Original vermerken
                dups = original[4]
                if row["path"] != original[2]["path"] and row["path"] not in dups:
                    dups.append(row["path"])
                seen_docs.add(row["document_id"])
                continue
        seen_docs.add(row["document_id"])
        kept.append((chunk_id, score, row, words, []))
        if len(kept) >= limit:
            break

    results: list[SearchHit] = []
    for chunk_id, score, row, _, dups in kept:
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
                duplicates=tuple(dups),
            )
        )
    return results


def merge_filters(a: SearchFilter, b: SearchFilter) -> SearchFilter:
    def pick(x: float | None, y: float | None, fn) -> float | None:
        return y if x is None else (x if y is None else fn(x, y))

    return SearchFilter(
        types=tuple(dict.fromkeys(a.types + b.types)),
        include=a.include + b.include,
        exclude=a.exclude + b.exclude,
        after=pick(a.after, b.after, max),
        before=pick(a.before, b.before, min),
        errors=a.errors + b.errors,
    )


def _filtered_chunk_ids(db: Database, flt: SearchFilter) -> list[int]:
    where, params = flt.sql("d")
    rows = db.conn.execute(
        f"SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id WHERE 1{where}", params
    )
    return [int(r[0]) for r in rows]


def _rerank(
    reranker: Reranker,
    query: str,
    fused: list[tuple[int, float]],
    rows: dict[int, dict],
    depth: int,
    blend: float | None = None,
) -> list[tuple[int, float]]:
    """Spitze der Liste neu bewerten; der Rest bleibt in alter Reihenfolge dahinter."""
    head = [(i, s) for i, s in fused[:depth] if i in rows]
    if not head:
        return fused
    scores = reranker.score(query, [_rerank_text(rows[i]) for i, _ in head])
    reranked = sorted(zip((i for i, _ in head), scores, strict=True), key=lambda x: x[1], reverse=True)
    if blend is not None:
        # Ränge mischen statt ersetzen: ein klarer Platz 1 der ersten Stufe fällt nicht so leicht heraus
        reranked = rrf([[i for i, _ in head], [i for i, _ in reranked]], weights=[1.0, blend])
    return reranked + fused[depth:]


def _rerank_text(row: dict) -> str:
    """Dateiname und Überschrift mitgeben – sie tragen oft das Thema, das im Abschnitt fehlt."""
    name = row["path"].replace("\\", "/").rsplit("/", 1)[-1]
    head = f"{name} – {row['heading_path']}" if row["heading_path"] else name
    return head + "\n" + row["text"]


_WORD = re.compile(r"[^\W_]+")
NEAR_DUPLICATE_JACCARD = 0.85
MIN_WORDS_FOR_FUZZY = 8


def _word_set(text: str) -> frozenset[str]:
    return frozenset(w.casefold() for w in _WORD.findall(text))


def _near_duplicate(a: frozenset[str], b: frozenset[str]) -> bool:
    """Gleich oder fast gleich (Jaccard über Wortmengen). Kurze Abschnitte nur bei exakt gleichen Wörtern."""
    if a == b:
        return bool(a)
    if min(len(a), len(b)) < MIN_WORDS_FOR_FUZZY:
        return False
    return len(a & b) / len(a | b) >= NEAR_DUPLICATE_JACCARD


def _fetch_chunks(db: Database, ids: list[int]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for start in range(0, len(ids), 500):  # SQLite-Parameterlimit
        batch = ids[start : start + 500]
        rows = db.conn.execute(
            "SELECT c.id, c.document_id, c.text, c.page, c.heading_path, d.path, d.filetype, d.mtime "
            "FROM chunks c JOIN documents d ON d.id = c.document_id "
            f"WHERE c.id IN ({','.join('?' * len(batch))})",
            batch,
        )
        out.update({int(r["id"]): dict(r) for r in rows})
    return out
