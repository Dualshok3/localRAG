import json
from pathlib import Path

import httpx
import pytest

from localrag.generation.answer import AnswerResult, extract_citations, retrieve_sources, stream_answer
from localrag.generation.llm import ChatClient, LLMError, LLMUnavailableError, ThinkFilter
from localrag.generation.prompts import NO_ANSWER, build_messages, select_context, source_label
from localrag.index.db import Database
from localrag.index.embedder import HashingEmbedder
from localrag.ingest.indexer import index_paths
from localrag.retrieval.keyword import SearchHit


def sse(*pieces: str) -> bytes:
    lines = [f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}" for p in pieces]
    return ("\n\n".join([*lines, "data: [DONE]"]) + "\n\n").encode()


class FakeServer:
    """Simuliert LM Studio: /v1/models, /api/v0/models, /v1/chat/completions (Streaming)."""

    def __init__(self, pieces: list[str], loaded: list[str] | None = None, lmstudio: bool = True) -> None:
        self.pieces = pieces
        self.loaded = loaded or []
        self.lmstudio = lmstudio  # False = anderer OpenAI-kompatibler Server ohne /api/v0
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            ids = ["text-embedding-nomic", "gemma-x", "qwen-y"]
            return httpx.Response(200, json={"data": [{"id": i} for i in ids]})
        if request.url.path == "/api/v0/models" and self.lmstudio:
            return httpx.Response(
                200, json={"data": [{"id": m, "state": "loaded", "type": "llm"} for m in self.loaded]}
            )
        if request.url.path == "/v1/chat/completions":
            self.requests.append(json.loads(request.content))
            return httpx.Response(
                200, content=sse(*self.pieces), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(404)


def client_for(server: FakeServer, model: str | None = None) -> ChatClient:
    return ChatClient("http://test/v1", model, transport=httpx.MockTransport(server))


def hit(text: str, path: str = r"D:\Docs\vertrag.pdf", page: int | None = 2, heading: str | None = None):
    return SearchHit(1, 1, path, "pdf", page, heading, text, 1.0, text[:50], ())


# --- ThinkFilter ----------------------------------------------------------------------------------


def test_think_filter_removes_reasoning_split_across_chunks():
    f = ThinkFilter()
    out = "".join(f.feed(c) for c in ["Vor<th", "ink>geheim", "es Denken</thi", "nk>Antwort [1]"]) + f.flush()
    assert out == "VorAntwort [1]"


def test_think_filter_passes_plain_text_and_lone_brackets():
    f = ThinkFilter()
    assert "".join(f.feed(c) for c in ["a < b", " und <", "x>"]) + f.flush() == "a < b und <x>"


# --- Client ---------------------------------------------------------------------------------------


def test_stream_chat_yields_content_and_sends_model():
    server = FakeServer(["Drei ", "Monate [1]."])
    with client_for(server, "qwen-y") as client:
        assert client.chat([]) == "Drei Monate [1]."
    assert server.requests[0]["model"] == "qwen-y"
    assert server.requests[0]["stream"] is True


def test_model_prefers_loaded_lmstudio_model():
    with client_for(FakeServer([], loaded=["qwen-y"])) as client:
        assert client.model == "qwen-y"


def test_model_falls_back_to_first_chat_model_on_other_servers():
    with client_for(FakeServer([], lmstudio=False)) as client:
        assert client.model == "gemma-x"  # Embedding-Modelle werden übersprungen


def test_lmstudio_without_loaded_model_is_an_error_not_a_silent_jit_load():
    # Realer Fall: Gemma war nicht geladen → LM Studio lud ungefragt das erste Modell der Liste
    with client_for(FakeServer([])) as client, pytest.raises(LLMError, match="lms load"):
        _ = client.model


def test_unreachable_server_gives_helpful_error():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with (
        ChatClient("http://test/v1", "m", transport=httpx.MockTransport(refuse)) as client,
        pytest.raises(LLMUnavailableError, match="lms server start"),
    ):
        client.chat([])


def test_server_error_is_reported():
    with (
        ChatClient("http://test/v1", "m", transport=httpx.MockTransport(lambda r: httpx.Response(500))) as c,
        pytest.raises(LLMError, match="500"),
    ):
        c.chat([])


# --- Prompt & Zitate ------------------------------------------------------------------------------


def test_source_label_and_prompt_numbering():
    sources = [hit("Kündigungsfrist drei Monate"), hit("Kaution 2100", page=None, heading="§4 Kaution")]
    assert source_label(sources[0]) == "vertrag.pdf · Seite 2"
    msgs = build_messages("Wie lange?", sources)
    assert msgs[0].role == "system" and "[1]" in msgs[0].content
    assert "[1] vertrag.pdf · Seite 2\nKündigungsfrist" in msgs[1].content
    assert "[2] vertrag.pdf · §4 Kaution" in msgs[1].content
    assert msgs[1].content.endswith("Frage: Wie lange?")


def test_select_context_respects_budget_but_keeps_first():
    big = [hit("x" * 8000), hit("y" * 8000), hit("z" * 100)]
    assert len(select_context(big, max_chars=10_000)) == 1
    assert len(select_context([hit("x" * 20_000)], max_chars=10_000)) == 1


def test_extract_citations_dedupes_and_ignores_out_of_range():
    assert extract_citations("A [2] B [1][2] C [9] D [0]", n_sources=3) == [2, 1]


# --- Ende-zu-Ende ---------------------------------------------------------------------------------


def test_answer_pipeline_end_to_end(db: Database, corpus: Path):
    emb = HashingEmbedder()
    index_paths(db, [corpus], embedder=emb)
    server = FakeServer(["Die Frist beträgt ", "drei Monate zum Monatsende [1]."])

    sources = retrieve_sources(db, "Kündigungsfrist", embedder=emb, mode="keyword")
    result = AnswerResult("Kündigungsfrist", sources)
    with client_for(server, "m") as client:
        text = "".join(stream_answer(client, result))

    assert text == result.text == "Die Frist beträgt drei Monate zum Monatsende [1]."
    assert Path(result.cited_sources[0][1].path).name == "Mietvertrag_2021.pdf"
    prompt = server.requests[0]["messages"][1]["content"]
    assert "Kündigungsfrist beträgt drei Monate" in prompt


def test_no_sources_means_no_llm_call():
    server = FakeServer(["sollte nie kommen"])
    result = AnswerResult("Frage", sources=[])
    with client_for(server, "m") as client:
        assert "".join(stream_answer(client, result)) == NO_ANSWER
    assert server.requests == []


def test_empty_model_output_becomes_no_answer():
    # Echter Fall mit Gemma 4: bei unbeantwortbarer Frage kam gar kein Text zurück
    result = AnswerResult("Frage", sources=[hit("x")])
    with client_for(FakeServer(["", "  "]), "m") as client:
        assert "".join(stream_answer(client, result)).strip() == NO_ANSWER
    assert result.is_no_answer and result.text == NO_ANSWER


def test_no_answer_is_recognised_even_with_extra_words():
    result = AnswerResult(
        "Frage", sources=[hit("x")], text="Leider: dazu habe ich in deinen Dateien nichts gefunden."
    )
    assert result.is_no_answer
    assert not AnswerResult("Frage", sources=[hit("x")], text="Drei Monate [1].").is_no_answer
