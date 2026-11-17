from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    """Ein zusammenhängender Textabschnitt eines Dokuments (z. B. eine PDF-Seite oder ein Kapitel)."""

    text: str
    page: int | None = None
    heading_path: str | None = None


class ParseError(Exception):
    """Datei konnte nicht gelesen werden (kaputt, verschlüsselt, falsches Format)."""


class HeadingStack:
    """Hält die aktuelle Überschriften-Hierarchie, z. B. 'Mietvertrag > §5 Kündigung'."""

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []

    def push(self, level: int, title: str) -> None:
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, title.strip()))

    @property
    def path(self) -> str | None:
        return " > ".join(t for _, t in self._stack) or None
