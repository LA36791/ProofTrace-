"""Tests for the AI evidence gate and evidence-backed reasoning."""

import pytest

from server.app.verifier import (
    LLMError,
    analyze,
    conclude,
    gate,
)


def _evidence(n=3, prefix="ev"):
    return [
        {
            "evidence_id": f"{prefix}_{i}",
            "file_path": f"file{i}.py",
            "line_start": 1,
            "line_end": 4,
            "text": f"def fn{i}(): return {i}",
        }
        for i in range(n)
    ]


def test_gate_sufficient_with_enough_evidence():
    verdict = gate(_evidence(3), "how does fn1 work?")
    assert verdict["outcome"] == "SUFFICIENT"
    assert verdict["missing"] == []


def test_gate_insufficient_with_abstention():
    verdict = gate(_evidence(1), "how does fn1 work?")
    assert verdict["outcome"] == "INSUFFICIENT"
    assert isinstance(verdict["missing"], list)
    assert len(verdict["missing"]) > 0


def test_gate_insufficient_with_no_evidence():
    verdict = gate([], "anything")
    assert verdict["outcome"] == "INSUFFICIENT"
    assert verdict["missing"] == ["No repository evidence was retrieved for this query."]


def test_conclude_cites_valid_evidence_ids():
    result = conclude(_evidence(3), "how does fn1 work?")
    assert result["statement"].strip()
    assert set(result["evidence_ids"]) <= {"ev_0", "ev_1", "ev_2"}
    assert len(result["evidence_ids"]) >= 1


def test_conclude_rejects_invalid_citation():
    evidence = _evidence(2)
    # Force a fabricated citation via a stub that returns an invalid id.
    class Stub:
        def chat(self, system, user, temperature=0.0):
            return '{"statement": "x", "evidence_ids": ["ev_999"]}'

    with pytest.raises(LLMError):
        conclude(evidence, "q", client=Stub())


def test_analyze_sufficient_generates_conclusion():
    result = analyze(_evidence(3), "how does fn1 work?")
    assert result["outcome"] == "SUFFICIENT"
    assert result["conclusion"] is not None
    assert result["conclusion"]["evidence_ids"]


def test_analyze_insufficient_abstains():
    result = analyze(_evidence(1), "how does fn1 work?")
    assert result["outcome"] == "INSUFFICIENT"
    assert result["conclusion"] is None
    assert len(result["missing"]) > 0


def test_analyze_no_evidence_abstains():
    result = analyze([], "anything")
    assert result["outcome"] == "INSUFFICIENT"
    assert result["conclusion"] is None
