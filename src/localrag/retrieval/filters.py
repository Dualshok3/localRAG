"""Filter direkt in der Suchanfrage: ``type:pdf``, ``in:Rechnungen``, ``-in:Backup``, ``after:2024-01``.

Beispiele
    Stromrechnung type:pdf after:2024
    Kündigungsfrist Mietvertrag -in:Backup
    Vertrag in:"Meine Dokumente" before:2023-06-30

- ``type:`` Dateityp, mehrere mit Komma (``type:pdf,docx``)
- ``in:`` / ``-in:`` Pfad enthält (nicht) diesen Ordner- oder Namensteil, ohne Groß-/Kleinschreibung
- ``after:`` / ``before:`` Änderungsdatum der Datei: ``2024``, ``2024-03`` oder ``2024-03-15``
  (``after:2024-03`` = ab 1. März, ``before:2024-03`` = vor 1. März)

Unbekannte Präfixe bleiben Teil der Suche – „C:\\Daten“ oder „Uhrzeit 10:30“ werden nicht zerlegt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import datetime

_TOKEN = re.compile(r'(?<!\S)(-?)(type|in|after|before):(?:"([^"]+)"|(\S+))', re.IGNORECASE)
_DATE = re.compile(r"^(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?$")


@dataclass(frozen=True)
class SearchFilter:
    types: tuple[str, ...] = ()
    include: tuple[str, ...] = ()  # Pfad muss jeden dieser Teile enthalten
    exclude: tuple[str, ...] = ()  # Pfad darf keinen davon enthalten
    after: float | None = None  # mtime >= after
    before: float | None = None  # mtime < before
    errors: tuple[str, ...] = field(default=(), compare=False)

    @property
    def active(self) -> bool:
        return bool(self.types or self.include or self.exclude or self.after or self.before)

    def with_types(self, types: list[str] | None) -> SearchFilter:
        if not types:
            return self
        merged = tuple(dict.fromkeys([*self.types, *(_norm_type(t) for t in types)]))
        return replace(self, types=merged)

    def matches(self, path: str, filetype: str, mtime: float) -> bool:
        if self.types and filetype not in self.types:
            return False
        folded = path.casefold().replace("/", "\\")
        if any(part not in folded for part in self.include):
            return False
        if any(part in folded for part in self.exclude):
            return False
        if self.after is not None and mtime < self.after:
            return False
        return not (self.before is not None and mtime >= self.before)

    def sql(self, doc_alias: str = "d") -> tuple[str, list[object]]:
        """WHERE-Bedingungen (mit führendem AND) für die Tabelle documents."""
        parts: list[str] = []
        params: list[object] = []
        if self.types:
            parts.append(f"{doc_alias}.filetype IN ({','.join('?' * len(self.types))})")
            params.extend(self.types)
        path = f"replace(casefold({doc_alias}.path), '/', '\\')"  # casefold ist in Database registriert
        for part in self.include:
            parts.append(f"instr({path}, ?) > 0")
            params.append(part)
        for part in self.exclude:
            parts.append(f"instr({path}, ?) = 0")
            params.append(part)
        if self.after is not None:
            parts.append(f"{doc_alias}.mtime >= ?")
            params.append(self.after)
        if self.before is not None:
            parts.append(f"{doc_alias}.mtime < ?")
            params.append(self.before)
        return "".join(f" AND {p}" for p in parts), params

    def describe(self) -> str:
        bits = []
        if self.types:
            bits.append("type:" + ",".join(self.types))
        bits += [f"in:{p}" for p in self.include] + [f"-in:{p}" for p in self.exclude]
        if self.after is not None:
            bits.append("after:" + datetime.fromtimestamp(self.after).strftime("%Y-%m-%d"))
        if self.before is not None:
            bits.append("before:" + datetime.fromtimestamp(self.before).strftime("%Y-%m-%d"))
        return " ".join(bits)


def _norm_type(t: str) -> str:
    return t.strip().lower().lstrip(".")


def _norm_path_part(p: str) -> str:
    return p.strip().strip("\\/").casefold().replace("/", "\\")


def parse_date(value: str, *, end: bool) -> float | None:
    """Beginn des Zeitraums als Unix-Zeit; ``end=True`` = Beginn des *folgenden* Zeitraums."""
    m = _DATE.match(value.strip())
    if not m:
        return None
    year, month, day = int(m.group(1)), m.group(2), m.group(3)
    try:
        if day:
            start = datetime(year, int(month), int(day))
            if end:
                return start.timestamp() + 86400
            return start.timestamp()
        if month:
            mo = int(month)
            if end:
                return datetime(year + (mo == 12), mo % 12 + 1, 1).timestamp()
            return datetime(year, mo, 1).timestamp()
        return datetime(year + 1 if end else year, 1, 1).timestamp()
    except ValueError:
        return None


def parse_query(query: str) -> tuple[str, SearchFilter]:
    """Trennt Filter-Angaben von der eigentlichen Suchanfrage."""
    types: list[str] = []
    include: list[str] = []
    exclude: list[str] = []
    after: float | None = None
    before: float | None = None
    errors: list[str] = []

    def take(m: re.Match[str]) -> str:
        nonlocal after, before
        negated, key, value = m.group(1) == "-", m.group(2).lower(), m.group(3) or m.group(4)
        if key == "type" and not negated:
            types.extend(_norm_type(t) for t in value.split(",") if t.strip())
        elif key == "in":
            part = _norm_path_part(value)
            if part:
                (exclude if negated else include).append(part)
        elif key in ("after", "before") and not negated:
            # after:2024 = ab Jahresbeginn; before:2024 = vor Jahresbeginn
            ts = parse_date(value, end=False)
            if ts is None:
                errors.append(f"Datum nicht erkannt: {key}:{value} (Format 2024, 2024-03 oder 2024-03-15)")
            elif key == "after":
                after = ts if after is None else max(after, ts)
            else:
                before = ts if before is None else min(before, ts)
        else:
            return m.group(0)
        return ""

    rest = _TOKEN.sub(take, query)
    clean = " ".join(rest.split())
    flt = SearchFilter(
        types=tuple(dict.fromkeys(types)),
        include=tuple(include),
        exclude=tuple(exclude),
        after=after,
        before=before,
        errors=tuple(errors),
    )
    return clean, flt
