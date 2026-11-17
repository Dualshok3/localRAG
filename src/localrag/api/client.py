"""Client für den lokalen Dienst – die CLI nutzt ihn automatisch, wenn er läuft."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

from localrag.api.server import hit_from_dict, server_url
from localrag.retrieval.keyword import SearchHit


class ServiceClient:
    def __init__(self, url: str | None = None, *, transport: httpx.BaseTransport | None = None) -> None:
        self.url = (url or server_url()).rstrip("/")
        self._http = httpx.Client(timeout=httpx.Timeout(300.0, connect=0.5), transport=transport)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ServiceClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def health(self) -> dict[str, Any] | None:
        """Status des Dienstes oder None, wenn er nicht läuft."""
        try:
            resp = self._http.get(f"{self.url}/health", timeout=1.0)
            return resp.json() if resp.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

    def serves(self, db_path: Path) -> dict[str, Any] | None:
        """Health-Daten, wenn der Dienst läuft *und* denselben Index nutzt."""
        info = self.health()
        if not info:
            return None
        same = os.path.normcase(str(Path(info["db"]))) == os.path.normcase(str(Path(db_path).resolve()))
        return info if same else None

    def search(
        self,
        query: str,
        *,
        mode: str = "auto",
        limit: int = 10,
        filetypes: list[str] | None = None,
        chunks: bool = False,
    ) -> tuple[str, float, list[SearchHit]]:
        params: list[tuple[str, str | int | bool]] = [("q", query), ("mode", mode), ("limit", limit)]
        params += [("type", t) for t in filetypes or []]
        if chunks:
            params.append(("chunks", "true"))
        resp = self._http.get(f"{self.url}/search", params=params)
        if resp.status_code >= 400:
            raise RuntimeError(resp.json().get("detail", resp.text))
        data = resp.json()
        return data["mode"], data["took_ms"], [hit_from_dict(h) for h in data["hits"]]

    def ask(
        self,
        question: str,
        *,
        k: int = 6,
        mode: str = "auto",
        filetypes: list[str] | None = None,
        llm_model: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        body = {"question": question, "k": k, "mode": mode, "types": filetypes, "llm_model": llm_model}
        with self._http.stream("POST", f"{self.url}/ask", json=body) as resp:
            if resp.status_code >= 400:
                resp.read()
                raise RuntimeError(resp.json().get("detail", resp.text))
            for line in resp.iter_lines():
                if line.strip():
                    event = json.loads(line)
                    if event.get("type") == "sources":
                        event["sources"] = [hit_from_dict(h) for h in event["sources"]]
                    yield event
