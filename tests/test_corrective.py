from server.app.retrieval.corrective import correct_retrieval


def _chunk(
    evidence_id,
    file_path,
    text,
    score,
):
    return {
        "evidence_id": evidence_id,
        "file_path": file_path,
        "line_start": 1,
        "line_end": 10,
        "text": text,
        "score": score,
    }


def test_corrective_keeps_relevant_evidence():
    candidates = [
        _chunk(
            "E1",
            "orders.py",
            "calculate subtotal and apply discount",
            0.90,
        ),
        _chunk(
            "E2",
            "pricing.py",
            "compute tax from discounted amount",
            0.80,
        ),
        _chunk(
            "E3",
            "unrelated.py",
            "logging configuration and startup",
            0.10,
        ),
    ]

    result = correct_retrieval(
        candidates,
        "discount subtotal tax",
        top_k=3,
    )

    ids = [
        item["evidence_id"]
        for item in result["results"]
    ]

    assert "E1" in ids
    assert "E2" in ids
    assert "E3" not in ids


def test_corrective_deduplicates_evidence():
    candidates = [
        _chunk(
            "E1",
            "orders.py",
            "calculate order subtotal",
            0.90,
        ),
        _chunk(
            "E1",
            "orders.py",
            "calculate order subtotal",
            0.80,
        ),
    ]

    result = correct_retrieval(
        candidates,
        "order subtotal",
        top_k=5,
    )

    ids = [
        item["evidence_id"]
        for item in result["results"]
    ]

    assert ids.count("E1") == 1


def test_corrective_handles_empty_retrieval():
    result = correct_retrieval(
        [],
        "unknown failure",
    )

    assert result["status"] == "INSUFFICIENT"
    assert result["results"] == []


def test_corrective_prefers_file_diversity():
    candidates = [
        _chunk(
            "E1",
            "orders.py",
            "order subtotal discount",
            0.95,
        ),
        _chunk(
            "E2",
            "orders.py",
            "order discount pricing",
            0.94,
        ),
        _chunk(
            "E3",
            "pricing.py",
            "discount tax calculation",
            0.80,
        ),
    ]

    result = correct_retrieval(
        candidates,
        "order discount tax",
        top_k=2,
    )

    files = {
        item["file_path"]
        for item in result["results"]
    }

    assert len(files) == 2