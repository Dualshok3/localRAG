"""BM25-Stichwortsuche über SQLite FTS5."""

from __future__ import annotations

from dataclasses import dataclass

from localrag.index.db import Database
from localrag.index.text_norm import iter_tokens, stem


@dataclass(frozen=True)
class SearchHit:
    chunk_id: int
    document_id: int
    path: str
    filetype: str
    page: int | None
    heading_path: str | None
    text: str
    score: float  # höher = besser; Bedeutung je nach Modus (BM25, Kosinus, RRF)
    snippet: str
    matched_words: tuple[str, ...]
    # Pfade weiterer Dateien mit (fast) gleichem Abschnitt, z. B. Backups oder PDF + DOCX derselben Datei
    duplicates: tuple[str, ...] = ()


def build_match_query(terms: list[str]) -> str:
    # Jeder Term in Anführungszeichen → FTS5-Sonderzeichen/Operatoren in der Eingabe sind harmlos.
    return " OR ".join(f'"{t}"' for t in terms)


def keyword_ranking(
    db: Database, terms: list[str], *, k: int, filetypes: list[str] | None = None
) -> list[tuple[int, float]]:
    """(chunk_id, BM25-Score) absteigend nach Relevanz."""
    if not terms:
        return []
    sql = (
        "SELECT chunks_fts.rowid AS id, bm25(chunks_fts) AS rank FROM chunks_fts "
        "JOIN chunks c ON c.id = chunks_fts.rowid JOIN documents d ON d.id = c.document_id "
        "WHERE chunks_fts MATCH ?"
    )
    params: list[object] = [build_match_query(terms)]
    if filetypes:
        sql += f" AND d.filetype IN ({','.join('?' * len(filetypes))})"
        params.extend(filetypes)
    sql += " ORDER BY rank LIMIT ?"
    params.append(k)
    # FTS5-bm25 ist negativ: kleiner = besser → Vorzeichen drehen
    return [(int(r["id"]), -float(r["rank"])) for r in db.conn.execute(sql, params)]


def make_snippet(text: str, terms: list[str], width: int = 240) -> tuple[str, tuple[str, ...]]:
    """Ausschnitt rund um den ersten Treffer + die tatsächlich getroffenen Originalwörter."""
    term_set = set(terms)
    hits = [(tok, start, end) for tok, start, end in iter_tokens(text) if stem(tok) in term_set]
    matched = tuple(dict.fromkeys(tok for tok, _, _ in hits))
    flat = " ".join(text.split())
    if not hits:
        return (flat[:width] + ("…" if len(flat) > width else "")), matched

    first = hits[0][0]
    pos = max(0, flat.casefold().find(first.casefold()))
    start = max(0, pos - width // 3)
    end = min(len(flat), start + width)
    if start > 0:
        space = flat.find(" ", start)  # nicht mitten im Wort beginnen
        start = space + 1 if 0 <= space < pos else start
    snippet = ("…" if start > 0 else "") + flat[start:end] + ("…" if end < len(flat) else "")
    return snippet, matched


def keyword_search(
    db: Database,
    query: str,
    *,
    limit: int = 10,
    filetypes: list[str] | None = None,
    one_per_document: bool = True,
) -> list[SearchHit]:
    from localrag.retrieval.hybrid import search

    return search(
        db, query, mode="keyword", limit=limit, filetypes=filetypes, one_per_document=one_per_document
    )
