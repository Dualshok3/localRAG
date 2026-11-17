from __future__ import annotations

from pathlib import Path

import docx
import pymupdf
import pytest

from localrag.index.db import Database


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "index.db")
    yield database
    database.close()


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Erzeugt ein PDF zur Laufzeit – so liegen keine Binärdateien im Repo."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
    doc.save(path)
    doc.close()
    return path


def make_docx(path: Path, blocks: list[tuple[str, str]]) -> Path:
    """blocks: (art, text) mit art in {'h1', 'h2', 'p'}."""
    document = docx.Document()
    for kind, text in blocks:
        if kind.startswith("h"):
            document.add_heading(text, level=int(kind[1]))
        else:
            document.add_paragraph(text)
    document.save(str(path))
    return path


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    (root / "Wohnung").mkdir(parents=True)
    (root / "node_modules").mkdir()
    make_pdf(
        root / "Wohnung" / "Mietvertrag_2021.pdf",
        [
            "Mietvertrag zwischen Vermieter Huber und Mieterin Beispiel.",
            "§5 Kündigung. Die Kündigungsfrist beträgt drei Monate zum Monatsende.",
        ],
    )
    make_docx(
        root / "Arbeitsvertrag.docx",
        [
            ("h1", "Arbeitsvertrag"),
            ("p", "Zwischen der Firma Muster GmbH und dem Arbeitnehmer."),
            ("h2", "Urlaub"),
            ("p", "Der Arbeitnehmer erhält 30 Urlaubstage pro Kalenderjahr."),
        ],
    )
    (root / "notizen.md").write_text(
        "# Einkauf\n\nMilch, Brot, Äpfel\n\n## Baumarkt\n\nSchrauben und Dübel kaufen.\n", encoding="utf-8"
    )
    (root / "alt.txt").write_bytes("Größe der Wohnung: 72 Quadratmeter".encode("cp1252"))
    (root / "node_modules" / "ignored.md").write_text("Kündigungsfrist sollte nie gefunden werden")
    (root / "bild.png").write_bytes(b"\x89PNG fake")
    return root
