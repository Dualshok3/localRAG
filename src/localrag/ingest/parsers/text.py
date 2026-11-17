from __future__ import annotations

import re
from pathlib import Path

from .base import HeadingStack, ParseError, Section, SkippedFile

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def read_text(path: Path) -> str:
    """UTF-8 (mit/ohne BOM), UTF-16 (wie es viele Windows-Programme schreiben), sonst Windows-1252."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ParseError(f"Datei nicht lesbar: {exc}") from exc
    utf16 = _decode_utf16(raw)
    if utf16 is not None:
        return utf16
    if b"\x00" in raw[:4096]:
        raise ParseError("Sieht nach Binärdatei aus")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _decode_utf16(raw: bytes) -> str | None:
    """UTF-16 mit BOM, oder ohne BOM erkannt am Muster „jedes zweite Byte ist 0“ (ASCII-lastiger Text)."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            return None
    head = raw[:4096]
    if len(head) < 8:
        return None
    odd_zeros = head[1::2].count(0) / len(head[1::2])
    even_zeros = head[0::2].count(0) / len(head[0::2])
    candidates = (("utf-16-le", odd_zeros, even_zeros), ("utf-16-be", even_zeros, odd_zeros))
    for encoding, zeros, other in candidates:
        if zeros > 0.9 and other < 0.05:
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                return None
    return None


# Wort- und Datenlisten erkennen (Wörterbücher, Tokenizer-Vokabular, Übersetzungstabellen, Daten-Dumps).
# Sie sind für eine Suche in eigenen Dokumenten nur Rauschen: Beim Test über ganze Laufwerke verdrängten
# sie mit Tausenden Abschnitten echte Treffer. Nur für .txt – Markdown-Tabellen sehen ähnlich aus.
LIST_MIN_LINES = 300
LIST_MAX_MEDIAN_WORDS = 3
LIST_MAX_SENTENCE_SHARE = 0.05
DUMP_MIN_LINE_CHARS = 50_000


def data_file_reason(text: str) -> str | None:
    """Grund, warum der Text wie eine Wort-/Datenliste statt wie ein Dokument aussieht – sonst None."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if any(len(ln) >= DUMP_MIN_LINE_CHARS for ln in lines):
        return "Daten-Dump (extrem lange Zeile)"
    if len(lines) < LIST_MIN_LINES:
        return None
    word_counts = sorted(len(ln.split()) for ln in lines)
    median = word_counts[len(word_counts) // 2]
    sentences = sum(1 for ln in lines if ln.endswith((".", "!", "?", ":")) and len(ln.split()) >= 5)
    if median <= LIST_MAX_MEDIAN_WORDS and sentences / len(lines) <= LIST_MAX_SENTENCE_SHARE:
        return "Wort- oder Datenliste"
    return None


def parse_txt(path: Path) -> list[Section]:
    text = read_text(path)
    reason = data_file_reason(text)
    if reason:
        raise SkippedFile(reason)
    return [Section(text=text)] if text.strip() else []


def parse_markdown(path: Path) -> list[Section]:
    sections: list[Section] = []
    headings = HeadingStack()
    buffer: list[str] = []
    in_code = False

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append(Section(text=text, heading_path=headings.path))
        buffer.clear()

    for line in read_text(path).splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
        m = None if in_code else _MD_HEADING_RE.match(line)
        if m:
            flush()
            headings.push(len(m.group(1)), m.group(2))
        else:
            buffer.append(line)
    flush()
    return sections
