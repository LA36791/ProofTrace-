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
- avoid aggressive filtering that destroys recall
- preserve useful weaker candidates when they still match the query
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
        for token in _TOKEN_RE.findall(str(text).lower())
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

    Retrieval score remains the primary signal.

    Query coverage is a supporting signal.

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
    """Select evidence while softly preferring file diversity.

    Retrieval strength remains the primary consideration.

    The strongest candidate is always retained.

    Diversity is used only after strong candidates have been ranked,
    preventing source diversity from destroying retrieval recall.
    """

    if limit <= 0:
        return []

    if not chunks:
        return []

    if len(chunks) <= limit:
        return chunks[:limit]

    selected: list[dict] = []

    selected_ids: set[str] = set()
    used_files: set[str] = set()

    # ---------------------------------------------------------------
    # Pass 1: always retain the strongest candidate.
    # ---------------------------------------------------------------

    first = chunks[0]

    first_id = str(
        first.get(
            "evidence_id",
            "",
        )
    )

    if first_id:
        selected.append(
            first
        )

        selected_ids.add(
            first_id
        )

    first_file = str(
        first.get(
            "file_path",
            "",
        )
    )

    if first_file:
        used_files.add(
            first_file
        )

    if len(selected) >= limit:
        return selected

    # ---------------------------------------------------------------
    # Pass 2: softly prefer new evidence sources.
    # ---------------------------------------------------------------

    for chunk in chunks[1:]:
        if len(selected) >= limit:
            break

        evidence_id = str(
            chunk.get(
                "evidence_id",
                "",
            )
        )

        if (
            not evidence_id
            or evidence_id in selected_ids
        ):
            continue

        file_path = str(
            chunk.get(
                "file_path",
                "",
            )
        )

        if (
            file_path
            and file_path not in used_files
        ):
            selected.append(
                chunk
            )

            selected_ids.add(
                evidence_id
            )

            used_files.add(
                file_path
            )

    # ---------------------------------------------------------------
    # Pass 3: fill remaining positions by corrective ranking.
    # ---------------------------------------------------------------

    for chunk in chunks:
        if len(selected) >= limit:
            break

        evidence_id = str(
            chunk.get(
                "evidence_id",
                "",
            )
        )

        if (
            not evidence_id
            or evidence_id in selected_ids
        ):
            continue

        selected.append(
            chunk
        )

        selected_ids.add(
            evidence_id
        )

    return selected[:limit]


def _build_files(
    chunks: list[dict],
) -> list[str]:
    """Return unique evidence file paths."""
    return sorted(
        {
            str(
                chunk.get(
                    "file_path",
                    "",
                )
            )
            for chunk in chunks
            if chunk.get(
                "file_path"
            )
        }
    )


def _build_query_coverage(
    chunks: list[dict],
) -> float:
    """Return the strongest query coverage among selected evidence."""
    return max(
        (
            float(
                chunk.get(
                    "query_coverage",
                    0.0,
                )
            )
            for chunk in chunks
        ),
        default=0.0,
    )


def correct_retrieval(
    candidates: list[dict],
    query: str,
    top_k: int = 5,
    min_quality: float = 0.08,
    min_fallback_coverage: float = 0.10,
) -> dict:
    """Correct and rerank Hybrid RAG candidates.

    The corrective stage:

    1. removes duplicate evidence IDs
    2. calculates query-term coverage
    3. calculates corrective relevance
    4. removes clearly irrelevant candidates
    5. preserves weaker candidates only when they still have
       meaningful query overlap
    6. reranks evidence deterministically
    7. softly prefers evidence from different files
    8. reports trace information for evaluation

    ``min_fallback_coverage`` prevents unrelated low-scoring evidence
    from being reintroduced merely for recall.

    Example:

        Relevant weak evidence:
            quality < min_quality
            but query coverage >= min_fallback_coverage

        Clearly unrelated evidence:
            quality < min_quality
            and query coverage == 0

    The Evidence Gate remains responsible for deciding whether the
    selected evidence is actually sufficient to answer the query.
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
    # 3. Score every candidate.
    # ---------------------------------------------------------------

    scored: list[
        tuple[dict, float, float]
    ] = []

    for chunk in unique:
        quality, coverage = _quality_score(
            chunk,
            query_terms,
        )

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
    # 4. Rank all candidates.
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

    # ---------------------------------------------------------------
    # 5. Strong candidates.
    # ---------------------------------------------------------------

    strong = [
        item
        for item in scored
        if item[1] >= min_quality
    ]

    # ---------------------------------------------------------------
    # 6. Recall-preserving candidates.
    #
    # A weak candidate is eligible only if it has meaningful query
    # overlap. This prevents unrelated evidence from returning.
    # ---------------------------------------------------------------

    recall_candidates = [
        item
        for item in scored
        if (
            item[1] < min_quality
            and item[2] >= min_fallback_coverage
        )
    ]

    # ---------------------------------------------------------------
    # 7. If nothing passes the normal threshold, preserve only
    # meaningful query-matching evidence.
    # ---------------------------------------------------------------

    if not strong:
        if recall_candidates:
            fallback_count = min(
                top_k,
                len(recall_candidates),
            )

            selected = [
                item[0]
                for item in recall_candidates[
                    :fallback_count
                ]
            ]

            query_coverage = _build_query_coverage(
                selected
            )

            files = _build_files(
                selected
            )

            return {
                "results": selected,
                "status": "FALLBACK",
                "reason": (
                    "Corrective threshold rejected the strongest "
                    "candidates, but query-matching fallback "
                    "evidence was preserved for verifier review."
                ),
                "query_coverage": round(
                    query_coverage,
                    6,
                ),
                "files": files,
            }

        return {
            "results": [],
            "status": "INSUFFICIENT",
            "reason": (
                "Hybrid retrieval candidates did not contain "
                "meaningful query-relevant evidence."
            ),
            "query_coverage": 0.0,
            "files": [],
        }

    # ---------------------------------------------------------------
    # 8. Build candidate pool.
    #
    # Strong evidence first.
    #
    # Then add only weak evidence that has meaningful query overlap.
    # ---------------------------------------------------------------

    candidate_pool = [
        item[0]
        for item in strong
    ]

    if len(candidate_pool) < top_k:
        remaining = top_k - len(
            candidate_pool
        )

        candidate_pool.extend(
            item[0]
            for item in recall_candidates[
                :remaining
            ]
        )

    # ---------------------------------------------------------------
    # 9. Soft source diversity.
    # ---------------------------------------------------------------

    selected = _select_diverse(
        candidate_pool,
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

    query_coverage = _build_query_coverage(
        selected
    )

    files = _build_files(
        selected
    )

    # ---------------------------------------------------------------
    # 10. Determine whether correction changed retrieval.
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
        :len(selected_ids)
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

    selected_weak_count = sum(
        1
        for chunk in selected
        if float(
            chunk.get(
                "corrective_score",
                0.0,
            )
        ) < min_quality
    )

    # ---------------------------------------------------------------
    # 11. Final corrective status.
    # ---------------------------------------------------------------

    if (
        removed_count > 0
        or changed
        or selected_weak_count > 0
    ):
        status = "CORRECTED"

        if selected_weak_count > 0:
            reason = (
                f"Corrective retrieval retained "
                f"{len(selected)} candidate(s), prioritized "
                f"strong evidence, preserved "
                f"{selected_weak_count} lower-scoring but "
                f"query-relevant candidate(s), and softly "
                f"preferred evidence-source diversity."
            )
        else:
            reason = (
                f"Corrective retrieval retained "
                f"{len(selected)} strong candidate(s), "
                f"removed or reordered weak evidence, "
                f"and softly preferred evidence-source diversity."
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