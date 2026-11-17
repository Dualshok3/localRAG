import sys
from pathlib import Path

import pymupdf
import pytest

from localrag.index.db import Database
from localrag.ingest.indexer import OCR_TRIED, index_paths
from localrag.ingest.parsers.pdf import parse_pdf, text_quality
from localrag.retrieval.hybrid import search


class FakeOcr:
    name = "fake-ocr"

    def __init__(self, text: str = "Abschlusszeugnis Gesamtnote 1,7") -> None:
        self.text = text
        self.calls = 0

    def recognize(self, png: bytes) -> str:
        assert png.startswith(b"\x89PNG")
        self.calls += 1
        return self.text


def make_scan(path: Path, text: str = "Abschlusszeugnis Gesamtnote 1,7") -> Path:
    """PDF, dessen Seite nur ein Bild des Textes enthält – wie ein Scan, ohne Textebene."""
    src = pymupdf.open()
    page = src.new_page(width=595, height=842)
    page.insert_text((72, 120), text, fontsize=28)
    pix = page.get_pixmap(dpi=150)
    scan = pymupdf.open()
    scan.new_page(width=595, height=842).insert_image(pymupdf.Rect(0, 0, 595, 842), pixmap=pix)
    scan.save(path)
    return path


def test_scan_has_no_text_layer(tmp_path: Path):
    assert parse_pdf(make_scan(tmp_path / "scan.pdf")) == []


def test_ocr_fills_pages_without_text(tmp_path: Path):
    ocr = FakeOcr()
    sections = parse_pdf(make_scan(tmp_path / "scan.pdf"), ocr=ocr)
    assert [s.page for s in sections] == [1] and "Gesamtnote" in sections[0].text
    assert ocr.calls == 1


def test_pages_with_text_are_not_ocred(tmp_path: Path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Ganz normaler Text mit ausreichend vielen Zeichen.")
    doc.save(tmp_path / "text.pdf")
    ocr = FakeOcr()
    assert "normaler Text" in parse_pdf(tmp_path / "text.pdf", ocr=ocr)[0].text
    assert ocr.calls == 0


def test_previously_empty_scans_are_ocred_once(db: Database, tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    make_scan(docs / "Zeugnis.pdf")

    first = index_paths(db, [docs])  # ohne OCR → leer
    assert first.empty == 1 and not search(db, "Gesamtnote", mode="keyword")

    ocr = FakeOcr()
    second = index_paths(db, [docs], ocr=ocr)  # unverändert, aber jetzt mit OCR → neu lesen
    assert second.updated == 1 and ocr.calls == 1
    assert Path(search(db, "Gesamtnote", mode="keyword")[0].path).name == "Zeugnis.pdf"

    third = index_paths(db, [docs], ocr=ocr)
    assert third.unchanged == 1 and ocr.calls == 1  # kein zweites OCR


def test_ocr_without_result_is_not_retried(db: Database, tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    make_scan(docs / "leer.pdf")
    blank = FakeOcr(text="")
    index_paths(db, [docs], ocr=blank)
    assert db.get_document(str((docs / "leer.pdf").resolve())).error == OCR_TRIED
    calls = blank.calls  # bei leerem Ergebnis werden alle vier Drehungen probiert
    index_paths(db, [docs], ocr=blank)
    assert blank.calls == calls


def test_failed_files_are_retried_once_after_parser_upgrade(db: Database, tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notiz.md").write_text("Inhalt", encoding="utf-8")
    index_paths(db, [docs])
    # Simulieren: Datei scheiterte mit einer älteren Parser-Version
    db.conn.execute("UPDATE documents SET status='error', error='alt' WHERE path LIKE '%notiz.md'")
    db.set_meta("parser_version", "1")

    assert index_paths(db, [docs]).updated == 1
    assert db.get_document(str((docs / "notiz.md").resolve())).status == "ok"
    assert index_paths(db, [docs]).unchanged == 1  # nur einmal


def test_text_quality_separates_words_from_garbage():
    assert text_quality("Bachelor of Engineering, Hochschule Landshut") > 0.8
    assert text_quality("-c o O 3 o 00 a) x— O oo 02 c o") < 0.2


def make_rotated_scan(path: Path, text: str) -> Path:
    src = pymupdf.open()
    page = src.new_page(width=595, height=842)
    page.insert_text((72, 120), text, fontsize=28)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2).prerotate(90))  # wie quer eingescannt
    scan = pymupdf.open()
    scan.new_page(width=842, height=595).insert_image(pymupdf.Rect(0, 0, 842, 595), pixmap=pix)
    scan.save(path)
    return path


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-OCR gibt es nur unter Windows")
def test_windows_ocr_handles_rotated_scans(tmp_path: Path):
    pytest.importorskip("winrt.windows.media.ocr")
    from localrag.ingest.ocr import WindowsOcr

    scan = make_rotated_scan(tmp_path / "quer.pdf", "Urkunde Bachelor of Engineering")
    text = " ".join(s.text for s in parse_pdf(scan, ocr=WindowsOcr()))
    assert "Bachelor" in text and "Engineering" in text


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-OCR gibt es nur unter Windows")
def test_windows_ocr_reads_rendered_text(tmp_path: Path):
    pytest.importorskip("winrt.windows.media.ocr")
    from localrag.ingest.ocr import WindowsOcr

    sections = parse_pdf(make_scan(tmp_path / "scan.pdf", "Kündigungsfrist drei Monate"), ocr=WindowsOcr())
    text = " ".join(s.text for s in sections)
    assert "Monate" in text and "drei" in text


def test_already_indexed_word_lists_are_dropped_after_parser_upgrade(db: Database, tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "vokabeln.txt").write_text("\n".join(f"Wort{i}" for i in range(500)), encoding="utf-8")
    (docs / "notiz.txt").write_text("Ein ganz normaler Satz.", encoding="utf-8")
    index_paths(db, [docs])
    # Simulieren: Index stammt von einer Version, die Wortlisten noch indexierte
    path = str((docs / "vokabeln.txt").resolve())
    db.conn.execute("UPDATE documents SET status='ok', error=NULL, chunk_count=3 WHERE path=?", (path,))
    db.set_meta("parser_version", "2")

    stats = index_paths(db, [docs])
    assert stats.updated == 1 and stats.unchanged == 1  # nur die Liste wird neu gelesen
    assert db.get_document(path).status == "skipped"
