import pytest

from localrag.ingest.chunker import chunk_section
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
