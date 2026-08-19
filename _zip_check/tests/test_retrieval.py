"""Tests for lexical retrieval behavior."""

import asyncio

from server.app.ingest import load_chunks
from server.app.retrieval import lexical_search


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


def test_returns_ranked_results_with_all_fields():
    res = lexical_search(_corpus(), "card charge tax amount", top_k=3)
    assert len(res) >= 1
    first = res[0]
    for key in ("evidence_id", "file_path", "line_start", "line_end", "text", "score"):
        assert key in first
    assert first["evidence_id"] == "ev_pay"
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)


def test_top_k_respected():
    res = lexical_search(_corpus(), "card charge token amount", top_k=1)
    assert len(res) == 1
    assert res[0]["evidence_id"] == "ev_pay"


def test_no_match_returns_empty():
    res = lexical_search(_corpus(), "shareholder logo color annual report", top_k=3)
    assert res == []


def test_ranking_relevant_over_irrelevant():
    res = lexical_search(_corpus(), "apply tax rate to amount", top_k=3)
    assert res[0]["evidence_id"] == "ev_tax"


def test_retrieval_over_ingested_repo():
    async def run():
        chunks = await load_chunks()
        assert len(chunks) > 0
        res = lexical_search(chunks, "apply discount to order total", top_k=5)
        assert len(res) > 0
        paths = {r["file_path"] for r in res}
        assert "data/demo_repo/pricing.py" in paths

    asyncio.run(run())
