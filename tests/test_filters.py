import os
from datetime import datetime
from pathlib import Path

import pytest

from localrag.index.db import Database
from localrag.index.embedder import HashingEmbedder
from localrag.index.vector_store import enable_memory_cache
from localrag.ingest.indexer import index_paths
from localrag.retrieval.filters import SearchFilter, parse_date, parse_query
from localrag.retrieval.hybrid import search


def names(hits) -> list[str]:
    return [Path(h.path).name for h in hits]


def ts(*args: int) -> float:
    return datetime(*args).timestamp()


def test_parse_query_extracts_filters_and_keeps_the_rest():
    q, f = parse_query('Stromrechnung type:pdf,DOCX in:Rechnungen -in:"Alte Backups" after:2024-03')
    assert q == "Stromrechnung"
    assert f.types == ("pdf", "docx")
    assert f.include == ("rechnungen",)
    assert f.exclude == ("alte backups",)
    assert f.after == ts(2024, 3, 1)
    assert f.before is None


@pytest.mark.parametrize(
    "query",
    ["Uhrzeit 10:30", r"Datei C:\Daten\x.pdf", "Mailto:info", "tin:foo", "Notiz: in Arbeit"],
)
def test_text_that_only_looks_like_a_filter_stays_in_the_query(query):
    q, f = parse_query(query)
    assert q == " ".join(query.split())
    assert not f.active


def test_dates_cover_whole_year_month_or_day():
    assert parse_date("2024", end=False) == ts(2024, 1, 1)
    assert parse_date("2024", end=True) == ts(2025, 1, 1)
    assert parse_date("2024-12", end=True) == ts(2025, 1, 1)
    assert parse_date("2024-02-29", end=True) == ts(2024, 3, 1)
    assert parse_date("2024-13", end=False) is None
    assert parse_date("gestern", end=False) is None


def test_invalid_date_is_reported_not_silently_dropped():
    q, f = parse_query("Vertrag after:letztes-jahr")
    assert q == "Vertrag"
    assert not f.active
    assert "after:letztes-jahr" in f.errors[0]


def test_matches_is_case_insensitive_and_handles_umlauts():
    f = SearchFilter(include=("überweisungen",), exclude=("backup",))
    assert f.matches(r"D:\Bank\Überweisungen\2024.pdf", "pdf", 0)
    assert not f.matches(r"D:\Bank\Überweisungen\Backup\2024.pdf", "pdf", 0)
    assert not f.matches(r"D:\Bank\Daueraufträge.pdf", "pdf", 0)


@pytest.fixture
def filtered_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    files = {
        "Überweisungen/strom_2023.md": ("Stromrechnung Abschlag Stadtwerke", ts(2023, 5, 1)),
        "Überweisungen/strom_2024.md": ("Stromrechnung Abschlag Stadtwerke Nachzahlung", ts(2024, 5, 1)),
        "Backup/Überweisungen/strom_2024.md": (
            "Stromrechnung Abschlag Stadtwerke Nachzahlung",
            ts(2024, 5, 2),
        ),
        "Notizen/strom.txt": ("Stromrechnung prüfen und Abschlag anpassen", ts(2024, 6, 1)),
    }
    for rel, (text, mtime) in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        os.utime(path, (mtime, mtime))
    return root


@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
def test_filters_in_query_narrow_every_search_mode(db: Database, filtered_corpus: Path, mode: str):
    emb = HashingEmbedder()
    index_paths(db, [filtered_corpus], embedder=emb)

    def found(q: str) -> list[str]:
        hits = search(db, q, mode=mode, embedder=emb, collapse_duplicates=False)
        return sorted(Path(h.path).relative_to(filtered_corpus).as_posix() for h in hits)

    assert found("Stromrechnung type:txt") == ["Notizen/strom.txt"]
    assert found("Stromrechnung in:überweisungen -in:backup") == [
        "Überweisungen/strom_2023.md",
        "Überweisungen/strom_2024.md",
    ]
    assert found("Stromrechnung before:2024") == ["Überweisungen/strom_2023.md"]
    assert found("Stromrechnung after:2024-05-02 type:md") == ["Backup/Überweisungen/strom_2024.md"]
    assert found("Stromrechnung in:gibtesnicht") == []


def test_filter_finds_subset_that_is_not_among_global_nearest_neighbours(db: Database, tmp_path: Path):
    """Vektorsuche mit Filter sucht exakt in der Teilmenge statt global mit Nachfiltern."""
    root = tmp_path / "d"
    (root / "viel").mkdir(parents=True)
    (root / "selten").mkdir()
    for i in range(60):
        (root / "viel" / f"n{i}.md").write_text(f"Stromrechnung Abschlag {i}", encoding="utf-8")
    (root / "selten" / "x.md").write_text("Rezept Lasagne", encoding="utf-8")
    emb = HashingEmbedder()
    index_paths(db, [root], embedder=emb)
    for memory in (False, True):
        enable_memory_cache(memory)
        try:
            hits = search(db, "Stromrechnung in:selten", mode="semantic", embedder=emb, limit=1, candidates=5)
        finally:
            enable_memory_cache(False)
        assert names(hits) == ["x.md"]


def test_parse_filters_can_be_turned_off(db: Database, filtered_corpus: Path):
    index_paths(db, [filtered_corpus])
    assert search(db, "Stromrechnung type:txt", mode="keyword", parse_filters=False)
    assert names(search(db, "Stromrechnung", mode="keyword", flt=SearchFilter(types=("txt",)))) == [
        "strom.txt"
    ]


def test_filters_are_not_sent_to_the_language_model():
    from localrag.generation.prompts import build_messages

    user = build_messages("Wie hoch war der Abschlag? in:Stadtwerke after:2024", [])[1].content
    assert user.endswith("Frage: Wie hoch war der Abschlag?")
