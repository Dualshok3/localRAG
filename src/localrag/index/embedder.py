"""Embedding-Modelle: Text → normalisierter Vektor.

Standard ist ``BAAI/bge-m3`` (mehrsprachig, 1024 Dimensionen). Das Modell wird beim ersten Aufruf
heruntergeladen (~2,3 GB). Mit ``LOCALRAG_MODELS`` lässt sich der Speicherort festlegen, z. B. auf D:.

Für Tests gibt es ``HashingEmbedder``: deterministisch, ohne PyTorch, aber ohne echtes Sprachverständnis.
"""

from __future__ import annotations

import hashlib
import os
from typing import Protocol

import numpy as np

from localrag.index.text_norm import iter_tokens, stem

DEFAULT_MODEL = "BAAI/bge-m3"
ENV_MODELS = "LOCALRAG_MODELS"
ENV_MODEL = "LOCALRAG_EMBEDDING_MODEL"

# Manche Modelle erwarten Präfixe für Anfrage und Dokument.
_PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
}


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SemanticUnavailableError(RuntimeError):
    pass


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        batch_size: int = 32,
        max_seq_length: int = 1024,
    ) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - abhängig von der Installation
            raise SemanticUnavailableError(
                "Semantische Suche nicht installiert. Ausführen: uv sync --extra semantic"
            ) from exc

        self.name = model_name or os.environ.get(ENV_MODEL, DEFAULT_MODEL)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        cache = os.environ.get(ENV_MODELS)
        self._model = SentenceTransformer(
            self.name,
            device=self.device,
            cache_folder=cache,
            model_kwargs={"torch_dtype": torch.float16} if self.device == "cuda" else None,
        )
        self._model.max_seq_length = min(max_seq_length, self._model.max_seq_length or max_seq_length)
        get_dim = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        self.dim = int(get_dim() or 0)
        self._query_prefix, self._doc_prefix = _PREFIXES.get(self.name, ("", ""))

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self._doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self._query_prefix + text])[0]


class HashingEmbedder:
    """Bag-of-words per Feature-Hashing. Nur für Tests und CI – versteht keine Synonyme."""

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok, _, _ in iter_tokens(text):
            h = int.from_bytes(hashlib.blake2b(stem(tok).encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


def load_embedder(model_name: str | None = None) -> Embedder:
    name = model_name or os.environ.get(ENV_MODEL, DEFAULT_MODEL)
    if name.startswith("hashing-"):
        return HashingEmbedder(int(name.split("-", 1)[1]))
    return SentenceTransformerEmbedder(name)


def semantic_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True
