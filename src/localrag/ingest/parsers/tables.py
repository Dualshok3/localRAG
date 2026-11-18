"""Tabellen: Excel (.xlsx/.xlsm) je Tabellenblatt, CSV als Ganzes.

Zeilen werden zu „Wert | Wert | Wert“. Bei Tabellen mit Kopfzeile kommt die Kopfzeile alle
``HEADER_EVERY`` Zeilen erneut in den Text – sonst wüsste ein Abschnitt aus der Mitte einer langen
Tabelle nicht mehr, welche Spalte „Betrag“ und welche „Datum“ ist.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
from pathlib import Path

from .base import ParseError, Section, SkippedFile
from .text import read_text

MAX_ROWS = 5_000  # pro Blatt; größere Tabellen sind meist Datenexporte, nicht Dokumente
HEADER_EVERY = 40


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dt.datetime):
        return (
            value.strftime("%d.%m.%Y %H:%M") if (value.hour or value.minute) else value.strftime("%d.%m.%Y")
        )
    if isinstance(value, dt.date):
        return value.strftime("%d.%m.%Y")
    if isinstance(value, float):
        return f"{value:.10g}".replace(".", ",")  # deutsche Schreibweise wie in den Dokumenten
    return str(value).strip()


def rows_to_text(rows: list[list[str]]) -> str:
    rows = [r for r in rows if any(c for c in r)]
    if not rows:
        return ""
    header = rows[0]
    has_header = len(rows) > 2 and all(header) and not any(c.replace(",", "").isdigit() for c in header)
    lines: list[str] = []
    for i, row in enumerate(rows):
        if has_header and i and i % HEADER_EVERY == 0:
            lines.append(" | ".join(header))
        # leere Zellen am Ende weglassen, in der Mitte als leere Spalte behalten
        while row and not row[-1]:
            row = row[:-1]
        lines.append(" | ".join(row))
    return "\n".join(lines)


def parse_xlsx(path: Path) -> list[Section]:
    import openpyxl

    try:
        # read_only: streamt große Dateien; data_only: berechnete Werte statt Formeln
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        raise ParseError(f"Excel-Datei nicht lesbar: {exc}") from exc
    sections: list[Section] = []
    try:
        for ws in wb.worksheets:
            rows: list[list[str]] = []
            for n, row in enumerate(ws.iter_rows(values_only=True)):
                if n >= MAX_ROWS:
                    break
                rows.append([_cell(v) for v in row])
            text = rows_to_text(rows)
            if text.strip():
                sections.append(Section(text=text, heading_path=f"Blatt {ws.title}"))
    finally:
        wb.close()
    return sections


def parse_csv(path: Path) -> list[Section]:
    text = read_text(path)
    sample = text[:8192]
    dialect: type[csv.Dialect] | csv.Dialect
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t|")
    except csv.Error:

        class Fallback(csv.excel):
            delimiter = ";" if sample.count(";") > sample.count(",") else ","

        dialect = Fallback
    rows = []
    for n, row in enumerate(csv.reader(io.StringIO(text), dialect)):
        if n >= MAX_ROWS:
            # Beim Test über ganze Laufwerke waren das Datensätze und Logs (62 MB Text aus 255 CSVs),
            # keine eigenen Dokumente – Kontoauszüge & Co. sind weit kürzer
            raise SkippedFile(f"Datenexport (über {MAX_ROWS} Zeilen)")
        rows.append([c.strip() for c in row])
    body = rows_to_text(rows)
    return [Section(text=body)] if body.strip() else []
