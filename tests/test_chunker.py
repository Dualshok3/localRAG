import pytest

from localrag.ingest.chunker import chunk_section, chunk_sections
from localrag.ingest.parsers.base import Section


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_short_text_is_single_chunk_and_keeps_metadata():
    chunks = chunk_section(Section("Hallo   Welt\n\nZweiter Absatz", page=3, heading_path="A > B"))
    assert len(chunks) == 1
    assert chunks[0].text == "Hallo Welt\n\nZweiter Absatz"
    assert (chunks[0].page, chunks[0].heading_path) == (3, "A > B")


def test_long_paragraph_is_windowed_with_overlap():
    chunks = chunk_section(Section(words(250)), max_words=100, overlap_words=20)
    assert all(c.word_count <= 100 for c in chunks)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-20:] == second[:20]
    # nichts geht verloren
    assert chunks[-1].text.split()[-1] == "w249"


def test_paragraphs_are_packed_and_overlap_reuses_tail_paragraph():
    paras = [words(40, f"p{i}_") for i in range(5)]
    chunks = chunk_section(Section("\n\n".join(paras)), max_words=100, overlap_words=45)
    assert all(c.word_count <= 100 for c in chunks)
    # Absatz 1 steht am Ende von Chunk 1 und am Anfang von Chunk 2
    assert chunks[0].text.endswith(paras[1])
    assert chunks[1].text.startswith(paras[1])


def test_empty_section_produces_no_chunks():
    assert chunk_section(Section("   \n\n  ")) == []


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_section(Section("x"), max_words=10, overlap_words=10)


def sentences(n: int, prefix: str) -> str:
    """n Sätze à 10 Wörter, jeder endet mit Punkt."""
    return " ".join(" ".join(f"{prefix}{i}_{j}" for j in range(9)) + f" {prefix}{i}_9." for i in range(n))


def test_long_last_paragraph_overlaps_by_sentences():
    # zwei Absätze à 60 Wörter: der erste passt nicht in 50 Wörter Überlappung → seine letzten Sätze
    text = sentences(6, "a") + "\n\n" + sentences(6, "b")
    chunks = chunk_section(Section(text), max_words=100, overlap_words=25)
    assert len(chunks) == 2
    second = chunks[1].text.split()
    # die letzten zwei ganzen Sätze (20 Wörter) des ersten Absatzes, beginnend am Satzanfang
    assert second[:20] == chunks[0].text.split()[40:60]
    assert second[0] == "a4_0"


def test_overlap_without_sentence_end_uses_last_words():
    text = words(60, "a") + "\n\n" + words(60, "b")
    chunks = chunk_section(Section(text), max_words=100, overlap_words=25)
    assert chunks[1].text.split()[:25] == [f"a{i}" for i in range(35, 60)]


def test_page_overlap_carries_end_of_previous_page():
    pages = [Section(sentences(3, "p1_"), page=1), Section(sentences(3, "p2_"), page=2)]
    plain = chunk_sections(pages, max_words=100, overlap_words=25)
    carried = chunk_sections(pages, max_words=100, overlap_words=25, page_overlap=True)
    assert plain[1].text.split()[0] == "p2_0_0"
    first_p2 = carried[1].text.split()
    assert first_p2[:20] == plain[0].text.split()[-20:]  # zwei ganze Sätze der Vorseite
    assert carried[1].page == 2  # Seitenangabe bleibt
    assert carried[0] == plain[0]


def test_page_overlap_only_for_consecutive_pages():
    pages = [Section(sentences(2, "x"), page=1), Section(sentences(2, "y"), page=3), Section("ohne Seite")]
    carried = chunk_sections(pages, page_overlap=True)
    assert carried[1].text.split()[0] == "y0_0"
    assert carried[2].text == "ohne Seite"
