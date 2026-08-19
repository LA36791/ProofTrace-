"""Tests for evaluation metrics and the end-to-end evaluation harness."""

import asyncio
import json

from server.app.evaluate.metrics import (
    abstention_correct,
    citation_valid,
    context_tokens,
    gate_accuracy,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from server.app.evaluate.run import evaluate_all, load_queries
from server.app.ingest import load_chunks


def _rel():
    return {"a", "b"}


def test_precision_at_k():
    assert precision_at_k(["a", "c", "b"], _rel(), 3) == 2 / 3
    assert precision_at_k(["c", "d"], _rel(), 3) == 0.0
    assert precision_at_k([], _rel(), 3) == 0.0
    assert precision_at_k(["a"], _rel(), 0) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "b"], _rel(), 5) == 1.0
    assert recall_at_k(["a"], _rel(), 5) == 0.5
    assert recall_at_k(["a"], _rel(), 1) == 0.5

    # Empty relevant set is not treated as perfect retrieval.
    assert recall_at_k([], set(), 3) == 0.0
    assert recall_at_k(["x"], set(), 3) == 0.0

    # Invalid k should return zero.
    assert recall_at_k(["a"], _rel(), 0) == 0.0


def test_ndcg_at_k():
    assert ndcg_at_k(["a", "b"], _rel(), 2) == 1.0
    assert ndcg_at_k(["c", "a"], _rel(), 2) < 1.0
    assert ndcg_at_k(["a", "b"], _rel(), 0) == 0.0

    # Empty relevant set is not treated as perfect ranking.
    assert ndcg_at_k(["x"], set(), 2) == 0.0
    assert ndcg_at_k([], set(), 2) == 0.0


def test_context_tokens():
    ev = [
        {"text": "x" * 40},
        {"text": "y" * 40},
    ]

    assert context_tokens(ev) == 20


def test_gate_accuracy():
    assert gate_accuracy("SUFFICIENT", True) is True
    assert gate_accuracy("SUFFICIENT", False) is False
    assert gate_accuracy("INSUFFICIENT", False) is True
    assert gate_accuracy("INSUFFICIENT", True) is False


def test_abstention_correct():
    assert abstention_correct(
        "INSUFFICIENT",
        False,
    ) is True

    assert abstention_correct(
        "INSUFFICIENT",
        True,
    ) is False

    assert abstention_correct(
        "SUFFICIENT",
        True,
    ) is True


def test_citation_valid():
    assert citation_valid(
        ["a", "b"],
        {"a", "b", "c"},
    ) is True

    assert citation_valid(
        ["a", "zzz"],
        {"a", "b"},
    ) is False

    assert citation_valid(
        [],
        {"a"},
    ) is False


def test_citation_valid_safe_abstention():
    assert citation_valid(
        [],
        set(),
        conclusion_generated=False,
        gate_outcome="INSUFFICIENT",
    ) is True


def test_citation_valid_rejects_unsafe_abstention():
    assert citation_valid(
        ["a"],
        {"a"},
        conclusion_generated=True,
        gate_outcome="INSUFFICIENT",
    ) is False


def test_citation_valid_accepts_evidence_backed_answer():
    assert citation_valid(
        ["a"],
        {"a", "b"},
        conclusion_generated=True,
        gate_outcome="SUFFICIENT",
    ) is True


def test_citation_valid_rejects_missing_citation():
    assert citation_valid(
        [],
        {"a", "b"},
        conclusion_generated=True,
        gate_outcome="SUFFICIENT",
    ) is False


def test_queries_json_loads():
    cases = load_queries()

    assert len(cases) >= 5

    for case in cases:
        assert "query" in case
        assert "sufficient" in case
        assert "relevant" in case


def test_end_to_end_evaluate_all():
    chunks = asyncio.run(
        load_chunks()
    )

    assert len(chunks) > 0

    report = evaluate_all(
        chunks
    )

    assert report["top_k"] > 0
    assert len(report["cases"]) >= 5

    for case in report["cases"]:
        for side in (
            "baseline",
            "current",
        ):
            record = case[side]

            assert isinstance(
                record["retrieved_ids"],
                list,
            )

            assert "precision_at_k" in record
            assert "recall_at_k" in record
            assert "ndcg_at_k" in record
            assert "citation_valid" in record

    json.dumps(report)