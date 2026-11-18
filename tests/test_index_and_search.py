import os
from pathlib import Path

from localrag.index.db import Database
from localrag.ingest.indexer import index_paths
from localrag.retrieval.keyword import keyword_search, make_snippet


def names(hits) -> list[str]:
    return [Path(h.path).name for h in hits]


def test_index_finds_supported_files_and_skips_excluded(db: Database, corpus: Path):
    stats = index_paths(db, [corpus])
    assert stats.added == 4  # pdf, docx, md, txt – nicht node_modules, nicht png
    assert stats.failed == 0
    assert stats.chunks >= 4


def test_search_with_german_inflection(db: Database, corpus: Path):
    index_paths(db, [corpus])
    hits = keyword_search(db, "Wie lang sind die Kündigungsfristen?")
    assert names(hits)[0] == "Mietvertrag_2021.pdf"
    assert hits[0].page == 2
    assert "Kündigungsfrist" in hits[0].matched_words


def test_search_matches_filename_and_headings(db: Database, corpus: Path):
    index_paths(db, [corpus])
    assert names(keyword_search(db, "mietvertrag"))[0] == "Mietvertrag_2021.pdf"
    hit = keyword_search(db, "Urlaubstage")[0]
    assert hit.heading_path == "Arbeitsvertrag > Urlaub"


def test_filetype_filter(db: Database, corpus: Path):
    index_paths(db, [corpus])
    assert keyword_search(db, "Kündigungsfrist Urlaub", filetypes=["docx"])[0].filetype == "docx"


def test_fts_special_characters_are_safe(db: Database, corpus: Path):
    index_paths(db, [corpus])
    assert keyword_search(db, 'Miet* "OR" NEAR( -') is not None
    assert keyword_search(db, "   ") == []


def test_incremental_update_and_delete(db: Database, corpus: Path):
    index_paths(db, [corpus])

    again = index_paths(db, [corpus])
    assert (again.added, again.updated, again.unchanged) == (0, 0, 4)

    note = corpus / "notizen.md"
    note.write_text("# Neu\n\nZahnarzttermin am Dienstag", encoding="utf-8")
    os.utime(note, (note.stat().st_atime, note.stat().st_mtime + 10))
    changed = index_paths(db, [corpus])
    assert changed.updated == 1
    assert names(keyword_search(db, "Zahnarzttermin")) == ["notizen.md"]
    assert keyword_search(db, "Schrauben Dübel") == []  # alter Inhalt ist weg

    note.unlink()
    removed = index_paths(db, [corpus])
    assert removed.removed == 1
    assert keyword_search(db, "Zahnarzttermin") == []
    fts_rows = db.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    chunk_rows = db.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert fts_rows == chunk_rows


def test_touched_but_unchanged_file_is_not_reparsed(db: Database, corpus: Path):
    index_paths(db, [corpus])
    f = corpus / "alt.txt"
    os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 100))
    stats = index_paths(db, [corpus])
    assert (stats.updated, stats.unchanged) == (0, 4)


def test_broken_file_is_recorded_not_fatal(db: Database, corpus: Path):
    (corpus / "kaputt.pdf").write_bytes(b"%PDF-1.4 kaputt")
    stats = index_paths(db, [corpus])
    assert stats.failed == 1 and stats.added == 5
    assert db.errors()[0][0].endswith("kaputt.pdf")


def test_indexing_other_root_does_not_delete_first_root(db: Database, corpus: Path, tmp_path: Path):
    index_paths(db, [corpus])
    other = tmp_path / "other"
    other.mkdir()
    (other / "x.md").write_text("anderes Dokument")
    stats = index_paths(db, [other])
    assert stats.removed == 0
    assert keyword_search(db, "Kündigungsfrist")


def test_snippet_centers_on_match():
    text = ("Füllwort " * 80) + "Hier steht die Kündigungsfrist von drei Monaten." + (" Ende" * 80)
    snippet, matched = make_snippet(text, ["kundigungsfrist"], width=120)
    assert "Kündigungsfrist" in snippet and snippet.startswith("…") and snippet.endswith("…")
    assert matched == ("Kündigungsfrist",)


def test_indexing_leaves_no_open_write_transaction(db: Database, corpus: Path):
    """Offene Transaktion = Schreibsperre: im Dienst blockierte das sonst Watcher und Uploads."""
    from localrag.index.embedder import HashingEmbedder

    index_paths(db, [corpus])
    assert not db.conn.in_transaction
    index_paths(db, [corpus], embedder=HashingEmbedder())  # nichts Neues einzubetten
    index_paths(db, [corpus], embedder=HashingEmbedder())
    assert not db.conn.in_transaction
