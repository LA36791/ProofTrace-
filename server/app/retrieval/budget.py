"""Context-budget selection over ranked evidence chunks.

Given a ranked list of chunks, greedily select complete chunks until a maximum
token budget is reached. Chunks are never truncated: if the next chunk would
exceed the budget it is skipped (and a skip is recorded) rather than cut.
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for source code."""
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def select_with_budget(
    ranked: list[dict],
    max_tokens: int,
    token_of: callable = estimate_tokens,
) -> dict:
    """Select complete evidence chunks within ``max_tokens``.

    Args:
        ranked: list of ranked chunk dicts (must include evidence_id and text).
        max_tokens: maximum total tokens allowed in the selected context.
        token_of: callable mapping chunk text -> estimated token count.

    Returns:
        dict with:
            selected_ids: list of evidence_ids kept (in ranked order).
            estimated_tokens: total estimated tokens of selected chunks.
            skipped_ids: list of evidence_ids excluded because they exceeded
                the remaining budget.
            exceeded: number of ranked chunks that were dropped.
    """
    if max_tokens <= 0:
        return {
            "selected_ids": [],
            "estimated_tokens": 0,
            "skipped_ids": [r["evidence_id"] for r in ranked],
            "exceeded": len(ranked),
        }

    selected_ids: list[str] = []
    skipped_ids: list[str] = []
    total = 0
    for chunk in ranked:
        est = token_of(chunk["text"])
        if total + est > max_tokens:
            skipped_ids.append(chunk["evidence_id"])
            continue
        selected_ids.append(chunk["evidence_id"])
        total += est

    return {
        "selected_ids": selected_ids,
        "estimated_tokens": total,
        "skipped_ids": skipped_ids,
        "exceeded": len(skipped_ids),
    }
