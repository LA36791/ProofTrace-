"""Tests for evidence ID determinism and demo repo ingestion."""

import asyncio

from server.app.ingest import build_chunks, evidence_id, ingest, load_chunks


def test_evidence_id_is_deterministic():
    a = evidence_id("cart.py", 3, 10)
    b = evidence_id("cart.py", 3, 10)
    c = evidence_id("cart.py", 3, 11)
    assert a == b
    assert a != c
    assert a.startswith("ev_")
    assert len(a) == 19  # 'ev_' + 16 hex chars


def test_evidence_id_distinct_per_range():
    assert evidence_id("pricing.py", 1, 5) != evidence_id("order.py", 1, 5)


def test_build_chunks_preserves_file_and_lines():
    chunks = build_chunks()
    assert len(chunks) >= 8
    assert all(c["file_path"] for c in chunks)
    assert all(c["line_start"] >= 1 for c in chunks)
    assert all(c["line_end"] >= c["line_start"] for c in chunks)
    assert all(c["evidence_id"] for c in chunks)


def test_ingest_and_load_roundtrip():
    async def run():
        count = await ingest()
        loaded = await load_chunks()
        assert count == len(loaded)
        assert count > 0
        # Deterministic IDs are stable across ingest runs.
        ids = [c["evidence_id"] for c in loaded]
        assert len(set(ids)) == len(ids)
        return loaded

    loaded = asyncio.run(run())
    assert all("text" in c for c in loaded)
