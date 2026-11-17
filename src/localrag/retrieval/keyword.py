"""BM25-Stichwortsuche über SQLite FTS5."""

from __future__ import annotations

from dataclasses import dataclass

from localrag.index.db import Database
from localrag.index.text_norm import iter_tokens, query_terms, stem


@dataclass(frozen=True)
class SearchHit:
    chunk_id: int
    document_id: int
    path: str
    filetype: str
    page: int | None
    heading_path: str | None
    text: str
    score: float  # höher = besser
    snippet: str
    matched_words: tuple[str, ...]


def build_match_query(terms: list[str]) -> str:
    # Jeder Term in Anführungszeichen → FTS5-Sonderzeichen/Operatoren in der Eingabe sind harmlos.
    return " OR ".join(f'"{t}"' for t in terms)


def make_snippet(text: str, terms: list[str], width: int = 240) -> tuple[str, tuple[str, ...]]:
    """Ausschnitt rund um den ersten Treffer + die tatsächlich getroffenen Originalwörter."""
    term_set = set(terms)
    hits = [(tok, start, end) for tok, start, end in iter_tokens(text) if stem(tok) in term_set]
    matched = tuple(dict.fromkeys(tok for tok, _, _ in hits))
    flat = " ".join(text.split())
    if not hits:
        return (flat[:width] + ("…" if len(flat) > width else "")), matched

    # Position im whitespace-normalisierten Text wiederfinden
    first = hits[0][0]
    pos = max(0, flat.casefold().find(first.casefold()))
    start = max(0, pos - width // 3)
    end = min(len(flat), start + width)
    if start > 0:
        # nicht mitten im Wort beginnen
        space = flat.find(" ", start)
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
    terms = query_terms(query)
    if not terms:
        return []

    sql = (
        "SELECT c.id, c.document_id, d.path, d.filetype, c.page, c.heading_path, c.text, "
        "bm25(chunks_fts) AS rank "
        "FROM chunks_fts "
        "JOIN chunks c ON c.id = chunks_fts.rowid "
        "JOIN documents d ON d.id = c.document_id "
        "WHERE chunks_fts MATCH ?"
    )
    params: list[object] = [build_match_query(terms)]
    if filetypes:
        types = [t.lower().lstrip(".") for t in filetypes]
        sql += f" AND d.filetype IN ({','.join('?' * len(types))})"
        params.extend(types)
    sql += " ORDER BY rank LIMIT ?"
    # Bei einem Treffer pro Dokument mehr Kandidaten holen, damit nach dem Deduplizieren genug übrig sind
    params.append(limit * 8 if one_per_document else limit)

    hits: list[SearchHit] = []
    seen_docs: set[int] = set()
    for row in db.conn.execute(sql, params):
        if one_per_document:
            if row["document_id"] in seen_docs:
                continue
            seen_docs.add(row["document_id"])
        snippet, matched = make_snippet(row["text"], terms)
        hits.append(
            SearchHit(
                chunk_id=row["id"],
                document_id=row["document_id"],
                path=row["path"],
                filetype=row["filetype"],
                page=row["page"],
                heading_path=row["heading_path"],
                text=row["text"],
                score=-float(row["rank"]),  # FTS5-bm25 ist negativ: kleiner = besser
                snippet=snippet,
                matched_words=matched,
            )
        )
        if len(hits) >= limit:
            break
    return hits
