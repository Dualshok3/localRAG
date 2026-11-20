"""Zerlegt Sections in Chunks fester Maximalgröße mit Überlappung.

Strategie:
1. Jede Section wird in Absätze (Leerzeilen) zerlegt, Whitespace normalisiert.
2. Zu lange Absätze werden per Wortfenster geteilt.
3. Absätze werden gierig zu Chunks bis ``max_words`` gepackt.
4. Ein neuer Chunk beginnt mit den letzten Absätzen des vorherigen (max. ``overlap_words`` Wörter),
   damit Informationen an Chunk-Grenzen nicht verloren gehen. Ist schon der letzte Absatz zu lang,
   werden stattdessen seine letzten Sätze übernommen (notfalls die letzten Wörter).
5. Bei Seiten (PDF) beginnt der erste Chunk einer Seite mit dem Ende der Vorseite, damit ein Satz über
   den Seitenumbruch nicht in zwei Hälften zerfällt.

Chunks gehören immer zu genau einer Section, damit Seite und Überschrift eindeutig bleiben.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from localrag.config import CHUNK_MAX_WORDS, CHUNK_OVERLAP_WORDS
from localrag.ingest.parsers.base import Section

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"[.!?…][\"'»«“”)\]]*$")

# Erhöhen, wenn sich die Aufteilung ändert → der Indexer zerlegt alle Dokumente einmal neu
CHUNKER_VERSION = 2  # 2: Satz-Überlappung bei langen Absätzen, Überlappung über PDF-Seiten


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int | None
    heading_path: str | None

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def _split_long(words: list[str], max_words: int, overlap: int) -> list[list[str]]:
    step = max(1, max_words - overlap)
    windows = []
    for start in range(0, len(words), step):
        windows.append(words[start : start + max_words])
        if start + max_words >= len(words):
            break
    return windows


def _tail_words(words: list[str], limit: int) -> list[str]:
    """Letzte ganze Sätze mit zusammen höchstens ``limit`` Wörtern, ohne Satzgrenze die letzten Wörter."""
    if limit <= 0 or not words:
        return []
    start = max(0, len(words) - limit)
    for i in range(max(0, start - 1), len(words) - 1):
        if _SENTENCE_END.search(words[i]):  # Satz endet bei i → ab i+1 folgen ganze Sätze
            return words[i + 1 :]
    return words[start:]


def _units(text: str, max_words: int, overlap: int) -> list[list[str]]:
    units: list[list[str]] = []
    for para in _PARAGRAPH_SPLIT.split(text):
        words = para.split()
        if not words:
            continue
        if len(words) > max_words:
            units.extend(_split_long(words, max_words, overlap))
        else:
            units.append(words)
    return units


def chunk_section(
    section: Section, max_words: int = CHUNK_MAX_WORDS, overlap_words: int = CHUNK_OVERLAP_WORDS
) -> list[Chunk]:
    if overlap_words >= max_words:
        raise ValueError("overlap_words muss kleiner als max_words sein")

    units = _units(section.text, max_words, overlap_words)
    chunks: list[Chunk] = []
    current: list[list[str]] = []
    current_words = 0

    def emit() -> None:
        text = "\n\n".join(" ".join(u) for u in current)
        chunks.append(Chunk(text=text, page=section.page, heading_path=section.heading_path))

    for unit in units:
        if current and current_words + len(unit) > max_words:
            emit()
            # Überlappung: hintere Absätze übernehmen, solange sie in overlap_words passen
            tail: list[list[str]] = []
            tail_words = 0
            for u in reversed(current):
                if tail_words + len(u) > overlap_words or tail_words + len(u) + len(unit) > max_words:
                    break
                tail.insert(0, u)
                tail_words += len(u)
            if not tail:  # schon der letzte Absatz ist zu lang → seine letzten Sätze übernehmen
                part = _tail_words(current[-1], min(overlap_words, max_words - len(unit)))
                if part:
                    tail, tail_words = [part], len(part)
            current, current_words = tail, tail_words
        current.append(unit)
        current_words += len(unit)

    if current:
        emit()
    return chunks


def chunk_sections(
    sections: list[Section],
    max_words: int = CHUNK_MAX_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
    *,
    page_overlap: bool = False,
) -> list[Chunk]:
    """Zerlegt alle Sections. Mit ``page_overlap`` beginnt jede Folgeseite mit dem Ende der Vorseite."""
    out: list[Chunk] = []
    prev: Section | None = None
    prev_last: Chunk | None = None
    for s in sections:
        chunks = chunk_section(s, max_words, overlap_words)
        if (
            page_overlap
            and chunks
            and prev is not None
            and prev_last is not None
            and prev.page is not None
            and s.page == prev.page + 1
        ):
            tail = _tail_words(prev_last.text.split(), overlap_words)
            if tail:
                chunks[0] = replace(chunks[0], text=" ".join(tail) + "\n\n" + chunks[0].text)
        out.extend(chunks)
        if chunks:
            prev, prev_last = s, chunks[-1]
    return out
