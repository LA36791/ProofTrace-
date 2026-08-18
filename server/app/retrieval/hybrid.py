"""Hybrid retrieval combining lexical (BM25) and semantic rankings via RRF.

Reciprocal Rank Fusion (RRF) merges multiple ranked lists deterministically:

    score(d) = sum over lists of 1 / (k + rank_d_in_list)

It is order-based, not score-based, so heterogeneous score scales (BM25 vs
cosine) are fused without renormalization. ``k=60`` is the standard default.
"""

from __future__ import annotations

from server.app.retrieval.lexical import lexical_search
from server.app.retrieval.semantic import semantic_search

_RRF_K = 60


def rrf_fuse(rankings: list[list[str]], k: int = _RRF_K) -> dict[str, float]:
    """Fuse ranked lists of evidence_ids into a single RRF score map.

    Args:
        rankings: each list is an evidence_id ranking (best first).
        k: RRF constant controlling rank weighting.

    Returns:
        dict mapping evidence_id -> fused RRF score.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, evidence_id in enumerate(ranking, start=1):
            fused[evidence_id] = fused.get(evidence_id, 0.0) + 1.0 / (k + rank)
    return fused


def hybrid_search(
    corpus: list[dict],
    query: str,
    top_k: int = 5,
    k: int = _RRF_K,
    embedder=None,
) -> list[dict]:
    """Combine BM25 and semantic rankings via RRF.

    Returns a ranked list of dicts with the standard retrieval fields plus
    ``rrf_score``, ``lexical_rank`` and ``semantic_rank`` for explainability.
    """
    lexical = lexical_search(corpus, query, top_k=len(corpus))
    semantic = semantic_search(corpus, query, top_k=len(corpus), embedder=embedder)

    lex_ids = [r["evidence_id"] for r in lexical]
    sem_ids = [r["evidence_id"] for r in semantic]
    fused = rrf_fuse([lex_ids, sem_ids], k=k)

    # Build lookup for per-source ranks.
    lex_rank = {eid: i + 1 for i, eid in enumerate(lex_ids)}
    sem_rank = {eid: i + 1 for i, eid in enumerate(sem_ids)}

    # Merge back with chunk metadata for the selected top_k.
    by_id = {c["evidence_id"]: c for c in corpus}
    ranked_ids = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for evidence_id, rrf_score in ranked_ids:
        if rrf_score <= 0:
            continue
        chunk = by_id.get(evidence_id)
        if chunk is None:
            continue
        results.append(
            {
                "evidence_id": evidence_id,
                "file_path": chunk["file_path"],
                "line_start": chunk["line_start"],
                "line_end": chunk["line_end"],
                "text": chunk["text"],
                "score": round(rrf_score, 6),
                "rrf_score": round(rrf_score, 6),
                "lexical_rank": lex_rank.get(evidence_id),
                "semantic_rank": sem_rank.get(evidence_id),
            }
        )
    return results
