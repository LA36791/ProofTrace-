"""Tests for hybrid retrieval, RRF fusion, and context budget."""

import pytest

from server.app.retrieval import (
    estimate_tokens,
    hybrid_search,
    lexical_search,
    rrf_fuse,
    select_with_budget,
)
from server.app.retrieval.semantic import FallbackSemanticEmbedder


def _corpus():
    return [
        {
            "evidence_id": "ev_cart",
            "file_path": "cart.py",
            "line_start": 1,
            "line_end": 5,
            "text": "CartItem computes line_total from unit_price times quantity",
        },
        {
            "evidence_id": "ev_tax",
            "file_path": "tax.py",
            "line_start": 1,
            "line_end": 4,
            "text": "compute_tax applies the TAX_RATE to an amount",
        },
        {
            "evidence_id": "ev_pay",
            "file_path": "payments.py",
            "line_start": 1,
            "line_end": 4,
            "text": "charge_card validates the token and charges a card amount",
        },
    ]


def test_lexical_still_works():
    res = lexical_search(_corpus(), "card charge", top_k=3)
    assert res[0]["evidence_id"] == "ev_pay"


def test_rrf_fuse_is_deterministic_and_correct():
    a = rrf_fuse([["x", "y", "z"], ["y", "x", "z"]])
    b = rrf_fuse([["x", "y", "z"], ["y", "x", "z"]])
    assert a == b
    assert a["x"] == pytest.approx(1 / 61 + 1 / 62)
    assert a["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert a["x"] == pytest.approx(a["y"])
    assert a["z"] == pytest.approx(2 / 63)


def test_hybrid_returns_deterministic_ranked_results():
    emb = FallbackSemanticEmbedder()
    a = hybrid_search(_corpus(), "card charge tax", top_k=3, embedder=emb)
    b = hybrid_search(_corpus(), "card charge tax", top_k=3, embedder=emb)
    assert [r["evidence_id"] for r in a] == [r["evidence_id"] for r in b]
    scores = [r["score"] for r in a]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_no_duplicate_evidence_ids():
    emb = FallbackSemanticEmbedder()
    res = hybrid_search(_corpus(), "tax card", top_k=3, embedder=emb)
    ids = [r["evidence_id"] for r in res]
    assert len(ids) == len(set(ids))


def test_top_k_respected_in_hybrid():
    emb = FallbackSemanticEmbedder()
    res = hybrid_search(_corpus(), "tax card charge", top_k=2, embedder=emb)
    assert len(res) <= 2


def test_no_match_remains_empty():
    emb = FallbackSemanticEmbedder()
    res = hybrid_search(
        _corpus(), "shareholder logo color annual report", top_k=3, embedder=emb
    )
    assert res == []


def test_context_budget_preserves_complete_chunks():
    ranked = [
        {"evidence_id": "a", "text": "x" * 100},
        {"evidence_id": "b", "text": "y" * 400},
        {"evidence_id": "c", "text": "z" * 40},
    ]
    budget = select_with_budget(ranked, max_tokens=50)
    assert "a" in budget["selected_ids"]
    assert "b" in budget["skipped_ids"]
    assert "c" in budget["selected_ids"]
    assert budget["estimated_tokens"] <= 50
    assert budget["exceeded"] == 1


def test_context_budget_respects_zero_budget():
    ranked = [{"evidence_id": "a", "text": "hello"}]
    budget = select_with_budget(ranked, max_tokens=0)
    assert budget["selected_ids"] == []
    assert budget["skipped_ids"] == ["a"]


def test_estimate_tokens():
    assert estimate_tokens("a" * 40) == 10
    assert estimate_tokens("") == 0
