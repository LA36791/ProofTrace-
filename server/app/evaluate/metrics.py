"""Evaluation metrics for retrieval, gating, abstention, and citations."""

from __future__ import annotations

import math

from server.app.retrieval.budget import estimate_tokens


def precision_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Fraction of top-k retrieved IDs that are relevant."""
    if k <= 0:
        return 0.0

    top = retrieved[:k]

    if not top:
        return 0.0

    hits = sum(
        1
        for evidence_id in top
        if evidence_id in relevant
    )

    return hits / len(top)


def recall_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Fraction of relevant IDs retrieved within top-k.

    Evaluation convention:
    when there are no labeled relevant documents, retrieval recall
    is reported as 0.0 rather than 1.0 so an empty retrieval does
    not appear as perfect retrieval performance.
    """
    if k <= 0:
        return 0.0

    if not relevant:
        return 0.0

    top = set(retrieved[:k])

    return len(top & relevant) / len(relevant)


def dcg(
    ranked: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Discounted cumulative gain at k using binary relevance."""
    score = 0.0

    for index, evidence_id in enumerate(
        ranked[:k],
        start=1,
    ):
        if evidence_id in relevant:
            score += 1.0 / math.log2(index + 1)

    return score


def ndcg_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
) -> float:
    """Normalized discounted cumulative gain at k.

    Evaluation convention:
    when there are no labeled relevant documents, nDCG is reported
    as 0.0 rather than 1.0 because there is no positive retrieval
    target against which to claim perfect ranking quality.
    """
    if k <= 0:
        return 0.0

    if not relevant:
        return 0.0

    actual = dcg(
        retrieved,
        relevant,
        k,
    )

    ideal = dcg(
        list(relevant),
        relevant,
        k,
    )

    if ideal == 0:
        return 0.0

    return actual / ideal


def context_tokens(
    evidence: list[dict],
) -> int:
    """Estimate total tokens in selected evidence."""
    return sum(
        estimate_tokens(
            item.get("text", "")
        )
        for item in evidence
    )


def gate_accuracy(
    outcome: str,
    ground_truth_sufficient: bool,
) -> bool:
    """Return whether the gate matched ground truth."""
    predicted = outcome == "SUFFICIENT"

    return predicted == ground_truth_sufficient


def abstention_correct(
    outcome: str,
    ground_truth_sufficient: bool,
) -> bool:
    """Return whether the gate made the correct allow/abstain decision."""
    return gate_accuracy(
        outcome,
        ground_truth_sufficient,
    )


def citation_valid(
    cited: list[str],
    retrieved_ids: set[str],
    conclusion_generated: bool | None = None,
    gate_outcome: str | None = None,
) -> bool:
    """Validate evidence attribution and safe abstention.

    Rules:

    1. Correct abstention:
       INSUFFICIENT + no conclusion = valid.

    2. Evidence-backed answer:
       SUFFICIENT + conclusion + valid citations = valid.

    3. Unsafe behavior:
       INSUFFICIENT + conclusion = invalid.

    4. Missing or invalid citations:
       conclusion exists but citations are missing/invalid = invalid.

    The optional arguments preserve compatibility with the original
    two-argument citation_valid() API.
    """

    # Backward-compatible behavior for callers that only provide
    # cited evidence IDs and retrieved IDs.
    if conclusion_generated is None:
        if not cited:
            return False

        return all(
            evidence_id in retrieved_ids
            for evidence_id in cited
        )

    # Correct abstention is a successful safety behavior.
    if (
        gate_outcome == "INSUFFICIENT"
        and not conclusion_generated
    ):
        return True

    # An insufficient gate must never produce a conclusion.
    if (
        gate_outcome == "INSUFFICIENT"
        and conclusion_generated
    ):
        return False

    # A sufficient answer must contain at least one citation.
    if conclusion_generated and not cited:
        return False

    # Every citation must point to retrieved evidence.
    if conclusion_generated:
        return all(
            evidence_id in retrieved_ids
            for evidence_id in cited
        )

    return False