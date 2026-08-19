"""Corrective retrieval stage for the Evidence Room pipeline.

Corrective RAG operates after Hybrid RAG.

Pipeline:

    Hybrid RAG
        ->
    Corrective RAG
        ->
    Context Budget
        ->
    Evidence Gate

Design goals:
- preserve strong Hybrid RAG candidates
- remove clearly irrelevant evidence
- improve query-term coverage
- prefer independent evidence sources when appropriate
- remain deterministic
- avoid an additional LLM call
- never fabricate evidence
"""

from __future__ import annotations

import re


_TOKEN_RE = re.compile(r"[a-z0-9_]+")

_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "for",
    "of",
    "to",
    "in",
    "on",
    "is",
    "are",
    "be",
    "was",
    "were",
    "with",
    "as",
    "at",
    "by",
    "that",
    "this",
    "what",
    "how",
    "does",
    "should",
    "why",
    "from",
    "it",
    "its",
}


def _tokenize(text: str) -> set[str]:
    """Return normalized non-stopword terms."""
    return {
        token
        for token in _TOKEN_RE.findall(
            str(text).lower()
        )
        if token not in _STOPWORDS
    }


def _overlap(
    query_terms: set[str],
    text: str,
) -> float:
    """Return the fraction of query terms found in evidence text."""
    if not query_terms:
        return 0.0

    evidence_terms = _tokenize(text)

    return len(
        query_terms & evidence_terms
    ) / len(query_terms)


def _score_value(
    chunk: dict,
) -> float:
    """Safely extract the original retrieval score."""
    try:
        return float(
            chunk.get(
                "score",
                0.0,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return 0.0


def _quality_score(
    chunk: dict,
    query_terms: set[str],
) -> tuple[float, float]:
    """Calculate deterministic corrective relevance.

    Retrieval score is the primary signal.

    Query coverage is the secondary signal.

    The resulting quality score is:

        70% retrieval score
        30% query-term coverage

    Returns:
        (quality_score, query_coverage)
    """

    retrieval_score = _score_value(
        chunk
    )

    coverage = _overlap(
        query_terms,
        str(
            chunk.get(
                "text",
                "",
            )
        ),
    )

    quality = (
        retrieval_score * 0.70
        + coverage * 0.30
    )

    return quality, coverage


def _deduplicate(
    chunks: list[dict],
) -> list[dict]:
    """Remove duplicate evidence IDs while keeping the strongest copy."""

    best: dict[str, dict] = {}

    for chunk in chunks:
        evidence_id = str(
            chunk.get(
                "evidence_id",
                "",
            )
        )

        if not evidence_id:
            continue

        existing = best.get(
            evidence_id
        )

        if existing is None:
            best[evidence_id] = chunk
            continue

        if (
            _score_value(chunk)
            > _score_value(existing)
        ):
            best[evidence_id] = chunk

    return list(
        best.values()
    )


def _select_diverse(
    chunks: list[dict],
    limit: int,
) -> list[dict]:
    """Select strong evidence while preferring file diversity.

    The first pass selects at most one strong candidate from each file.

    The second pass fills remaining positions with the strongest unused
    candidates.

    Example:

        E1 orders.py
        E2 orders.py
        E3 pricing.py

    with top_k=2 becomes:

        E1 + E3

    rather than:

        E1 + E2
    """

    if limit <= 0:
        return []

    selected: list[dict] = []

    used_ids: set[str] = set()
    used_files: set[str] = set()

    # ---------------------------------------------------------------
    # Pass 1: maximize source diversity.
    # ---------------------------------------------------------------

    for chunk in chunks:
        evidence_id = str(
            chunk.get(
                "evidence_id",
                "",
            )
        )

        file_path = str(
            chunk.get(
                "file_path",
                "",
            )
        )

        if not evidence_id:
            continue

        if evidence_id in used_ids:
            continue

        if file_path in used_files:
            continue

        selected.append(
            chunk
        )

        used_ids.add(
            evidence_id
        )

        used_files.add(
            file_path
        )

        if len(selected) >= limit:
            return selected

    # ---------------------------------------------------------------
    # Pass 2: fill remaining positions with strongest unused evidence.
    # ---------------------------------------------------------------

    for chunk in chunks:
        evidence_id = str(
            chunk.get(
                "evidence_id",
                "",
            )
        )

        if not evidence_id:
            continue

        if evidence_id in used_ids:
            continue

        selected.append(
            chunk
        )

        used_ids.add(
            evidence_id
        )

        if len(selected) >= limit:
            break

    return selected


def correct_retrieval(
    candidates: list[dict],
    query: str,
    top_k: int = 5,
    min_quality: float = 0.08,
) -> dict:
    """Correct and rerank Hybrid RAG candidates.

    The corrective stage:

    1. removes duplicates
    2. calculates query-term coverage
    3. calculates corrective relevance
    4. removes clearly weak candidates
    5. reranks surviving evidence
    6. prefers evidence from different files
    7. reports trace information for evaluation

    Args:
        candidates:
            Candidate evidence returned by Hybrid RAG.

        query:
            Original user investigation query.

        top_k:
            Maximum number of evidence chunks returned.

        min_quality:
            Minimum corrective relevance score.

            The default is intentionally calibrated to Hybrid RAG's
            Reciprocal Rank Fusion score scale. RRF scores are small
            values, so an aggressive threshold such as 0.20 would
            incorrectly reject valid evidence.

    Returns:
        {
            "results": [...],
            "status": "PASS" | "CORRECTED" | "INSUFFICIENT",
            "reason": "...",
            "query_coverage": float,
            "files": [...]
        }
    """

    # ---------------------------------------------------------------
    # 1. No candidates.
    # ---------------------------------------------------------------

    if not candidates:
        return {
            "results": [],
            "status": "INSUFFICIENT",
            "reason": (
                "Hybrid retrieval returned no candidates."
            ),
            "query_coverage": 0.0,
            "files": [],
        }

    query_terms = _tokenize(
        query
    )

    # ---------------------------------------------------------------
    # 2. Deduplicate evidence.
    # ---------------------------------------------------------------

    unique = _deduplicate(
        candidates
    )

    if not unique:
        return {
            "results": [],
            "status": "INSUFFICIENT",
            "reason": (
                "Hybrid retrieval returned no valid evidence IDs."
            ),
            "query_coverage": 0.0,
            "files": [],
        }

    # ---------------------------------------------------------------
    # 3. Score candidates.
    # ---------------------------------------------------------------

    scored: list[
        tuple[dict, float, float]
    ] = []

    for chunk in unique:
        quality, coverage = _quality_score(
            chunk,
            query_terms,
        )

        # Remove clearly irrelevant candidates.
        if quality < min_quality:
            continue

        corrected = dict(
            chunk
        )

        corrected[
            "retrieval_score"
        ] = round(
            _score_value(chunk),
            6,
        )

        corrected[
            "corrective_score"
        ] = round(
            quality,
            6,
        )

        corrected[
            "query_coverage"
        ] = round(
            coverage,
            6,
        )

        scored.append(
            (
                corrected,
                quality,
                coverage,
            )
        )

    # ---------------------------------------------------------------
    # 4. No candidate passed correction.
    # ---------------------------------------------------------------

    if not scored:
        return {
            "results": [],
            "status": "INSUFFICIENT",
            "reason": (
                "Hybrid retrieval returned candidates, but none "
                "passed the corrective relevance threshold."
            ),
            "query_coverage": 0.0,
            "files": [],
        }

    # ---------------------------------------------------------------
    # 5. Corrective reranking.
    #
    # Primary:
    #     corrective score
    #
    # Secondary:
    #     query coverage
    #
    # Tertiary:
    #     original retrieval score
    # ---------------------------------------------------------------

    scored.sort(
        key=lambda item: (
            item[1],
            item[2],
            _score_value(
                item[0]
            ),
        ),
        reverse=True,
    )

    corrected_candidates = [
        item[0]
        for item in scored
    ]

    # ---------------------------------------------------------------
    # 6. Select evidence with source diversity.
    # ---------------------------------------------------------------

    selected = _select_diverse(
        corrected_candidates,
        top_k,
    )

    if not selected:
        return {
            "results": [],
            "status": "INSUFFICIENT",
            "reason": (
                "Corrective retrieval could not select reliable "
                "evidence."
            ),
            "query_coverage": 0.0,
            "files": [],
        }

    # ---------------------------------------------------------------
    # 7. Aggregate query coverage.
    #
    # Maximum coverage is used because one evidence chunk may fully
    # answer one important aspect of a query.
    # ---------------------------------------------------------------

    query_coverage = max(
        float(
            chunk.get(
                "query_coverage",
                0.0,
            )
        )
        for chunk in selected
    )

    # ---------------------------------------------------------------
    # 8. Evidence source list.
    # ---------------------------------------------------------------

    files = sorted(
        {
            str(
                chunk.get(
                    "file_path",
                    "",
                )
            )
            for chunk in selected
            if chunk.get(
                "file_path"
            )
        }
    )

    # ---------------------------------------------------------------
    # 9. Determine whether correction changed retrieval.
    # ---------------------------------------------------------------

    original_ids = [
        chunk.get(
            "evidence_id"
        )
        for chunk in candidates
    ]

    selected_ids = [
        chunk.get(
            "evidence_id"
        )
        for chunk in selected
    ]

    original_top_ids = original_ids[
        : len(selected_ids)
    ]

    changed = (
        selected_ids
        != original_top_ids
    )

    removed_count = max(
        0,
        len(candidates)
        - len(selected),
    )

    # ---------------------------------------------------------------
    # 10. Final corrective status.
    # ---------------------------------------------------------------

    if (
        removed_count > 0
        or changed
    ):
        status = "CORRECTED"

        reason = (
            f"Corrective retrieval retained "
            f"{len(selected)} strong candidate(s), "
            f"removed or reordered weak evidence, "
            f"and applied evidence-source diversity."
        )

    else:
        status = "PASS"

        reason = (
            "Hybrid retrieval candidates passed corrective "
            "relevance checks without requiring changes."
        )

    return {
        "results": selected,
        "status": status,
        "reason": reason,
        "query_coverage": round(
            query_coverage,
            6,
        ),
        "files": files,
    }