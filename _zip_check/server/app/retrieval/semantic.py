"""Semantic retrieval over evidence chunks.

Two providers behind one interface:

- ``ApiSemanticEmbedder``: calls an OpenAI-compatible embeddings API (via
  httpx). Active only when ``EMBEDDING_API_KEY`` is configured in the env.
- ``FallbackSemanticEmbedder``: a dependency-free, deterministic hashing-based
  embedder. Used when no API credentials are present. It is NOT a learned
  semantic model; treat it as a stand-in so the pipeline and tests run.

Real semantic retrieval is active only when the API provider is used.
"""

from __future__ import annotations

import hashlib
import os
import statistics
from abc import ABC, abstractmethod

import httpx

from server.app.retrieval.lexical import tokenize

# Fixed-dimension hashing embedding (fallback only). 384 is a common small
# embedding size and keeps memory negligible for the demo corpus.
_DIM = 384

# API config (read from .env via the existing config approach).
_API_KEY = os.getenv("EMBEDDING_API_KEY", "").strip()
_API_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1").rstrip("/")
_API_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


class SemanticEmbedder(ABC):
    """Interface every embedder must implement."""

    name: str

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a fixed-size embedding vector for ``text``."""

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class FallbackSemanticEmbedder(SemanticEmbedder):
    """Deterministic, dependency-free hashing embedder (NOT a learned model)."""

    name = "fallback-hashing"

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * _DIM
        for token in tokenize(text):
            idx = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % _DIM
            vec[idx] += 1.0
        norm = sum(x * x for x in vec) ** 0.5
        if norm == 0:
            return vec
        return [x / norm for x in vec]


class ApiSemanticEmbedder(SemanticEmbedder):
    """OpenAI-compatible embeddings API client (real semantic embeddings)."""

    name = "api"

    def __init__(
        self,
        api_key: str,
        base_url: str = _API_BASE_URL,
        model: str = _API_MODEL,
    ) -> None:
        self._key = api_key
        self._base_url = base_url
        self._model = model
        self._client = httpx.Client(timeout=30)

    def embed(self, text: str) -> list[float]:
        resp = self._client.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._key}"},
            json={"model": self._model, "input": text},
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


def get_embedder() -> SemanticEmbedder:
    """Return the API embedder when configured, else the deterministic fallback."""
    if _API_KEY:
        return ApiSemanticEmbedder(_API_KEY)
    return FallbackSemanticEmbedder()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def semantic_search(
    corpus: list[dict], query: str, top_k: int = 5, embedder: SemanticEmbedder | None = None
) -> list[dict]:
    """Rank evidence chunks by embedding similarity to the query.

    Returns the same result shape as lexical ``rank_evidence`` but with a
    semantic ``score`` (cosine similarity in [-1, 1]).

    When the fallback (non-API) embedder is used, chunks that share no lexical
    tokens with the query are excluded: the hashing fallback is not a learned
    model, so cross-token hash collisions would otherwise produce spurious
    positive scores for genuinely unrelated chunks.
    """
    emb = embedder or get_embedder()
    use_overlap_gate = emb.name == "fallback-hashing"
    q_tokens = set(tokenize(query))
    q_vec = emb.embed(query)
    scored = []
    for chunk in corpus:
        if use_overlap_gate and not (q_tokens & set(tokenize(chunk["text"]))):
            continue
        c_vec = emb.embed(chunk["text"])
        scored.append((chunk, _cosine(q_vec, c_vec)))
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    results = []
    for chunk, score in ranked[:top_k]:
        if score <= 0:
            break
        results.append(
            {
                "evidence_id": chunk["evidence_id"],
                "file_path": chunk["file_path"],
                "line_start": chunk["line_start"],
                "line_end": chunk["line_end"],
                "text": chunk["text"],
                "score": round(score, 6),
            }
        )
    return results


def is_real_semantic() -> bool:
    """True when real (API) embeddings are configured and will be used."""
    return bool(_API_KEY)
