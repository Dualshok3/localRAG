"""Client für OpenAI-kompatible Chat-Server (LM Studio, llama.cpp, vLLM, Ollama …).

Standard ist der lokale Server von LM Studio unter http://localhost:1234/v1
(starten mit ``lms server start`` oder in LM Studio unter „Developer“).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

ENV_LLM_URL = "LOCALRAG_LLM_URL"
ENV_LLM_MODEL = "LOCALRAG_LLM_MODEL"
ENV_LLM_API_KEY = "LOCALRAG_LLM_API_KEY"
DEFAULT_LLM_URL = "http://localhost:1234/v1"


class LLMError(RuntimeError):
    pass


class LLMUnavailableError(LLMError):
    pass


@dataclass(frozen=True)
class Message:
    role: str  # system | user | assistant
    content: str


class ThinkFilter:
    """Entfernt <think>…</think>-Blöcke von Reasoning-Modellen aus einem Token-Stream.

    Tags können über mehrere Stream-Stücke verteilt ankommen, daher wird ein kleiner Puffer gehalten.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, chunk: str) -> str:
        self._buf += chunk
        out = []
        while self._buf:
            tag = self.CLOSE if self._inside else self.OPEN
            idx = self._buf.find(tag)
            if idx >= 0:
                if not self._inside:
                    out.append(self._buf[:idx])
                self._buf = self._buf[idx + len(tag) :]
                self._inside = not self._inside
                continue
            # Ende könnte ein angefangenes Tag sein → zurückhalten
            keep = next((n for n in range(len(tag) - 1, 0, -1) if self._buf.endswith(tag[:n])), 0)
            if not self._inside:
                out.append(self._buf[: len(self._buf) - keep])
            self._buf = self._buf[len(self._buf) - keep :] if keep else ""
            break
        return "".join(out)

    def flush(self) -> str:
        rest, self._buf = ("" if self._inside else self._buf), ""
        return rest


class ChatClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get(ENV_LLM_URL) or DEFAULT_LLM_URL).rstrip("/")
        self._model = model or os.environ.get(ENV_LLM_MODEL)
        key = api_key or os.environ.get(ENV_LLM_API_KEY) or "not-needed"
        self._http = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=5.0),
            headers={"Authorization": f"Bearer {key}"},
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ChatClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- Modelle -------------------------------------------------------------------------------

    def list_models(self) -> list[str]:
        data = self._get_json(f"{self.base_url}/models")
        return [m["id"] for m in data.get("data", [])]

    def _loaded_lmstudio_models(self) -> list[str]:
        """LM Studio kennt zusätzlich /api/v0/models mit Ladezustand. Andere Server: leere Liste."""
        root = re.sub(r"/v1$", "", self.base_url)
        try:
            data = self._get_json(f"{root}/api/v0/models")
        except LLMError:
            return []
        return [
            m["id"]
            for m in data.get("data", [])
            if m.get("state") == "loaded" and m.get("type") != "embeddings"
        ]

    @property
    def model(self) -> str:
        """Konfiguriertes Modell, sonst ein bereits geladenes, sonst das erste Chat-Modell."""
        if not self._model:
            loaded = self._loaded_lmstudio_models()
            candidates = loaded or [m for m in self.list_models() if "embed" not in m.lower()]
            if not candidates:
                raise LLMError("Der Server meldet kein Chat-Modell. In LM Studio ein Modell laden.")
            self._model = candidates[0]
        return self._model

    # --- Chat ----------------------------------------------------------------------------------

    def stream_chat(
        self, messages: list[Message], *, temperature: float = 0.2, max_tokens: int = 1024
    ) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        think = ThinkFilter()
        try:
            with self._http.stream("POST", f"{self.base_url}/chat/completions", json=payload) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    raise LLMError(f"Server-Fehler {resp.status_code}: {resp.text[:300]}")
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    for choice in event.get("choices", []):
                        # reasoning_content / reasoning (Denkprozess) bewusst ignorieren
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            visible = think.feed(text)
                            if visible:
                                yield visible
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(self._unavailable_hint()) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"Zeitüberschreitung beim LLM-Server {self.base_url}") from exc
        rest = think.flush()
        if rest:
            yield rest

    def chat(self, messages: list[Message], **kwargs: float | int) -> str:
        return "".join(self.stream_chat(messages, **kwargs))  # type: ignore[arg-type]

    # --- intern --------------------------------------------------------------------------------

    def _get_json(self, url: str) -> dict:
        try:
            resp = self._http.get(url)
        except httpx.ConnectError as exc:
            raise LLMUnavailableError(self._unavailable_hint()) from exc
        except httpx.HTTPError as exc:
            raise LLMError(str(exc)) from exc
        if resp.status_code >= 400:
            raise LLMError(f"{url} → HTTP {resp.status_code}")
        return resp.json()

    def _unavailable_hint(self) -> str:
        return (
            f"Kein LLM-Server unter {self.base_url} erreichbar. "
            "LM Studio starten und den Server aktivieren: `lms server start`"
        )
