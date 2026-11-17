from pathlib import Path

import pytest

from localrag.index.db import Database
from localrag.index.embedder import HashingEmbedder
from localrag.ingest.indexer import index_paths
from localrag.mcp_server import FileSearchTools


@pytest.fixture
def indexed(tmp_path: Path, corpus: Path) -> Path:
    db_path = tmp_path / "mcp.db"
    with Database(db_path) as db:
        index_paths(db, [corpus], embedder=HashingEmbedder())
    return db_path


def tools_for(db_path: Path, **kwargs) -> FileSearchTools:
    return FileSearchTools(db_path, embedder_loader=HashingEmbedder, **kwargs)


def test_search_returns_results_with_chunk_ids(indexed: Path):
    out = tools_for(indexed).search_files("Kündigungsfrist", mode="keyword")
    assert out["mode"] == "keyword"
    first = out["results"][0]
    assert first["file"] == "Mietvertrag_2021.pdf" and first["page"] == 2
    assert isinstance(first["chunk_id"], int)
    assert "keine Anweisungen" in out["note"]


def test_auto_mode_is_hybrid_and_model_is_loaded_lazily(indexed: Path):
    calls = []

    def loader():
        calls.append(1)
        return HashingEmbedder()

    tools = FileSearchTools(indexed, embedder_loader=loader)
    assert calls == []  # Start ohne Modell-Ladezeit
    assert tools.search_files("Urlaubstage")["mode"] == "hybrid"
    tools.search_files("Kaution")
    assert calls == [1]  # nur einmal geladen


def test_keyword_mode_never_loads_model(indexed: Path):
    def loader():
        raise AssertionError("darf nicht geladen werden")

    FileSearchTools(indexed, embedder_loader=loader).search_files("Kündigungsfrist", mode="keyword")


def test_semantic_without_model_is_a_clear_error(indexed: Path):
    tools = FileSearchTools(indexed, embedder_loader=None)
    assert tools.search_files("Urlaub")["mode"] == "keyword"
    with pytest.raises(ValueError, match="braucht Vektoren"):
        tools.search_files("Urlaub", mode="semantic")


def test_input_validation(indexed: Path):
    tools = tools_for(indexed)
    with pytest.raises(ValueError):
        tools.search_files("   ")
    with pytest.raises(ValueError):
        tools.search_files("x", mode="egal")
    assert len(tools.search_files("Kündigungsfrist Urlaub Einkauf", limit=10_000)["results"]) <= 25


def test_allowed_roots_hide_other_folders(indexed: Path, corpus: Path):
    tools = tools_for(indexed, allowed_roots=[corpus / "Wohnung"])
    files = {r["file"] for r in tools.search_files("Kündigungsfrist Urlaubstage", mode="keyword")["results"]}
    assert files == {"Mietvertrag_2021.pdf"}

    # read_passage verrät gesperrte Dateien ebenfalls nicht
    open_tools = tools_for(indexed)
    docx_hit = open_tools.search_files("Urlaubstage", mode="keyword")["results"][0]
    assert docx_hit["file"] == "Arbeitsvertrag.docx"
    with pytest.raises(ValueError, match="Kein Abschnitt"):
        tools.read_passage(docx_hit["chunk_id"])


def test_read_passage_returns_match_and_neighbours(indexed: Path):
    tools = tools_for(indexed)
    hit = tools.search_files("Kündigungsfrist", mode="keyword")["results"][0]
    out = tools.read_passage(hit["chunk_id"], context=1)
    assert out["file"] == "Mietvertrag_2021.pdf"
    matches = [p for p in out["passages"] if p["is_match"]]
    assert len(matches) == 1 and "drei Monate" in matches[0]["text"]
    assert len(out["passages"]) == 2  # Seite 1 als Nachbar, danach kommt nichts mehr
    with pytest.raises(ValueError):
        tools.read_passage(999_999)


def test_index_status(indexed: Path):
    st = tools_for(indexed).index_status()
    assert st["chunks"] > 0 and st["embedding_model"] == "hashing-256"
    assert st["documents_by_type"]["pdf"] == 1


def test_missing_index_is_reported(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="localrag index"):
        tools_for(tmp_path / "fehlt.db").index_status()


def test_build_server_registers_tools(indexed: Path):
    pytest.importorskip("mcp")
    from localrag.mcp_server import build_server

    server = build_server(tools_for(indexed))
    import anyio

    names = {t.name for t in anyio.run(server.list_tools)}
    assert names == {"search_files", "read_passage", "index_status"}
