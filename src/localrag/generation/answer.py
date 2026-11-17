"""RAG-Pipeline: suchen → Kontext auswählen → LLM fragen → Zitate auflösen."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from localrag.generation.llm import ChatClient
from localrag.generation.prompts import NO_ANSWER, build_messages, select_context
from localrag.index.db import Database
from localrag.index.embedder import Embedder
from localrag.retrieval.hybrid import Mode, search
from localrag.retrieval.keyword import SearchHit

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


@dataclass
class AnswerResult:
    question: str
    sources: list[SearchHit]
    text: str = ""
    cited: list[int] = field(default_factory=list)  # 1-basierte Quellennummern in Reihenfolge

    @property
    def is_no_answer(self) -> bool:
        """Modell hat (wie angewiesen) gesagt, dass die Quellen nichts hergeben."""
        return NO_ANSWER.rstrip(".").casefold() in self.text.casefold()

    @property
    def cited_sources(self) -> list[tuple[int, SearchHit]]:
        return [(n, self.sources[n - 1]) for n in self.cited if 0 < n <= len(self.sources)]


def extract_citations(text: str, n_sources: int) -> list[int]:
    seen: list[int] = []
    for m in _CITATION_RE.finditer(text):
        n = int(m.group(1))
        if 1 <= n <= n_sources and n not in seen:
            seen.append(n)
    return seen


def retrieve_sources(
    db: Database,
    question: str,
    *,
    embedder: Embedder | None,
    mode: Mode = "auto",
    k: int = 6,
    filetypes: list[str] | None = None,
) -> list[SearchHit]:
    # Mehrere Abschnitte derselben Datei sind hier erwünscht (z. B. Kapitel 3 und 5 eines Vertrags)
    hits = search(
        db, question, mode=mode, embedder=embedder, limit=k, filetypes=filetypes, one_per_document=False
    )
    return select_context(hits)


def stream_answer(
    client: ChatClient, result: AnswerResult, *, temperature: float = 0.2, max_tokens: int = 1024
) -> Iterator[str]:
    """Streamt die Antwort und füllt ``result.text`` und ``result.cited``."""
    if not result.sources:
        result.text = NO_ANSWER
        yield NO_ANSWER
        return
    parts: list[str] = []
    for piece in client.stream_chat(
        build_messages(result.question, result.sources), temperature=temperature, max_tokens=max_tokens
    ):
        parts.append(piece)
        yield piece
    result.text = "".join(parts).strip()
    result.cited = extract_citations(result.text, len(result.sources))
