"""Reranker: bewertet Frage und Textabschnitt gemeinsam (Cross-Encoder) und sortiert die Kandidaten neu.

Die erste Stufe (BM25 + Vektoren) findet schnell ~50 Kandidaten, vergleicht aber nur Wörter bzw. zwei
getrennt berechnete Vektoren. Ein Cross-Encoder liest Frage und Abschnitt zusammen und erkennt so
Umschreibungen ("Wohnung kündigen" ↔ "Kündigungsfrist der Mieterin"). Er ist zu langsam für den ganzen Index,
aber schnell genug für die Spitze der Liste.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Protocol

from localrag.index.embedder import ENV_MODELS, SemanticUnavailableError

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
ENV_RERANKER = "LOCALRAG_RERANKER"


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Relevanz je Text, höher = besser. Nur die Reihenfolge zählt."""
        ...


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - abhängig von der Installation
            raise SemanticUnavailableError(
                "Reranker nicht installiert. Ausführen: uv sync --extra semantic"
            ) from exc

        self.name = model_name or os.environ.get(ENV_RERANKER, DEFAULT_RERANKER)
        from localrag.config import resolve_device

        self.device = resolve_device(device) or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._model = CrossEncoder(
            self.name,
            device=self.device,
            max_length=max_length,
            cache_folder=os.environ.get(ENV_MODELS),
            model_kwargs={"torch_dtype": torch.float16} if self.device == "cuda" else None,
        )

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        scores = self._model.predict(
            [(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]
