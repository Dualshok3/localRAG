from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from localrag.api.client import ServiceClient
from localrag.api.server import create_app
from localrag.generation.llm import ChatClient
from localrag.index.db import Database
from localrag.index.embedder import HashingEmbedder
from localrag.ingest.indexer import index_paths

from .test_generation import FakeServer


class Bridge(httpx.BaseTransport):
    """Leitet httpx-Anfragen an den FastAPI-TestClient weiter (ohne echten Netzwerk-Port)."""

    def __init__(self, test_client: TestClient) -> None:
        self.test_client = test_client

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        resp = self.test_client.request(
            request.method, str(request.url), content=request.read(), headers=dict(request.headers)
        )
        return httpx.Response(resp.status_code, headers=resp.headers, content=resp.content)


@pytest.fixture
def indexed(tmp_path: Path, corpus: Path) -> Path:
    db_path = tmp_path / "svc.db"
    with Database(db_path) as db:
        index_paths(db, [corpus], embedder=HashingEmbedder())
    return db_path


def make_service(db_path: Path, llm: FakeServer | None = None, *, with_vectors: bool = True):
    def factory(model: str | None) -> ChatClient:
        return ChatClient("http://llm/v1", model or "m", transport=httpx.MockTransport(llm or FakeServer([])))

    app = create_app(
        db_path, embedder=HashingEmbedder() if with_vectors else None, load_model=False, llm_factory=factory
    )
    test_client = TestClient(app)
    return test_client, ServiceClient("http://testserver", transport=Bridge(test_client))


def test_health_reports_db_and_model(indexed: Path):
    _, svc = make_service(indexed)
    info = svc.health()
    assert info["status"] == "ok" and info["embedding_model"] == "hashing-256"
    assert svc.serves(indexed)
    assert svc.serves(indexed.parent / "anderer.db") is None


def test_health_is_none_when_service_is_down():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert ServiceClient("http://x", transport=httpx.MockTransport(refuse)).health() is None


def test_search_through_service_matches_local_search(indexed: Path):
    _, svc = make_service(indexed)
    mode, took_ms, hits = svc.search("Kündigungsfrist", mode="keyword")
    assert mode == "keyword" and took_ms >= 0
    assert Path(hits[0].path).name == "Mietvertrag_2021.pdf"
    assert hits[0].page == 2 and "Kündigungsfrist" in hits[0].matched_words


def test_auto_mode_uses_hybrid_when_model_loaded(indexed: Path):
    _, svc = make_service(indexed)
    assert svc.search("Urlaubstage")[0] == "hybrid"


def test_semantic_without_model_is_rejected(indexed: Path):
    _, svc = make_service(indexed, with_vectors=False)
    assert svc.search("Urlaub")[0] == "keyword"
    with pytest.raises(RuntimeError, match="braucht Vektoren"):
        svc.search("Urlaub", mode="semantic")


def test_search_filters_and_validation(indexed: Path):
    client, svc = make_service(indexed)
    _, _, hits = svc.search("Kündigungsfrist Urlaub", mode="keyword", filetypes=["docx"])
    assert {h.filetype for h in hits} == {"docx"}
    assert client.get("/search", params={"q": ""}).status_code == 422
    assert client.get("/search", params={"q": "x", "limit": 1000}).status_code == 422


def test_ask_streams_sources_tokens_and_citations(indexed: Path):
    llm = FakeServer(["Drei Monate ", "zum Monatsende [1]."])
    _, svc = make_service(indexed, llm)
    events = list(svc.ask("Kündigungsfrist", mode="keyword"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "sources" and kinds[1] == "model" and kinds[-1] == "done"
    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert text == "Drei Monate zum Monatsende [1]."
    assert events[-1]["cited"] == [1]
    assert Path(events[0]["sources"][0].path).name == "Mietvertrag_2021.pdf"


def test_ask_reports_llm_errors_as_event(indexed: Path):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    app = create_app(
        indexed,
        embedder=None,
        load_model=False,
        llm_factory=lambda m: ChatClient("http://llm/v1", "m", transport=httpx.MockTransport(refuse)),
    )
    test_client = TestClient(app)
    svc = ServiceClient("http://testserver", transport=Bridge(test_client))
    events = list(svc.ask("Kündigungsfrist"))
    assert events[-1]["type"] == "error" and "lms server start" in events[-1]["message"]


def test_open_connection_sees_files_indexed_later(indexed: Path, tmp_path: Path):
    client, _ = make_service(indexed, with_vectors=False)
    assert client.get("/search", params={"q": "Zitronenfalter"}).json()["hits"] == []

    extra = tmp_path / "neu"
    extra.mkdir()
    (extra / "garten.md").write_text("Im Garten flog ein Zitronenfalter.", encoding="utf-8")
    with Database(indexed) as db:  # wie ein zweiter Prozess, der während des Dienstes indexiert
        index_paths(db, [extra])

    hits = client.get("/search", params={"q": "Zitronenfalter"}).json()["hits"]
    assert [Path(h["path"]).name for h in hits] == ["garten.md"]


def test_service_uses_reranker_when_configured(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Reparatur Reparatur Reparatur am Fahrrad.", encoding="utf-8")
    (docs / "b.md").write_text("Reparatur der Heizung und des Thermostats.", encoding="utf-8")
    db_path = tmp_path / "rr.db"
    with Database(db_path) as db:
        index_paths(db, [docs])

    class PreferKotfluegel:
        name = "fake-reranker"

        def score(self, query, texts):
            return [1.0 if "Thermostat" in t else 0.0 for t in texts]

    def first(app) -> str:
        return Path(TestClient(app).get("/search", params={"q": "Reparatur"}).json()["hits"][0]["path"]).name

    assert first(create_app(db_path, load_model=False)) == "a.md"
    reranked = create_app(db_path, load_model=False, reranker=PreferKotfluegel())
    assert first(reranked) == "b.md"
    assert TestClient(reranked).get("/health").json()["reranker"] == "fake-reranker"
