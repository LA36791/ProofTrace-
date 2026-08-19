"""Evaluation harness comparing BASELINE vs CURRENT retrieval/reasoning.

Run:
    .venv/Scripts/python.exe -m server.app.evaluate.run

BASELINE:
    query -> lexical retrieval -> context budget -> reasoning (no gate)

CURRENT:
    query -> hybrid retrieval -> context budget -> gate
    -> conditional reasoning/abstention

Works hermetically without a real LLM. If OPENAI_API_KEY is set but invalid,
the LLM path degrades to the deterministic fallback and the run still
completes.

The evaluation distinguishes:
- retrieval quality
- gate accuracy
- correct abstention
- citation correctness
- context size
- retrieval latency
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from server.app.ingest import load_chunks
from server.app.retrieval import (
    hybrid_search,
    lexical_search,
    select_with_budget,
)
from server.app.verifier import (
    LLMClient,
    LLMError,
    analyze,
    conclude,
    is_llm_active,
)

from server.app.evaluate.metrics import (
    abstention_correct,
    citation_valid,
    context_tokens,
    gate_accuracy,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
QUERIES_PATH = PROJECT_ROOT / "eval" / "queries.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

TOP_K = 5
MAX_TOKENS = 800


def load_queries(path: Path = QUERIES_PATH) -> list[dict]:
    """Load the labeled evaluation queries."""

    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def relevant_ids_for(
    chunks: list[dict],
    case: dict,
) -> set[str]:
    """Map labeled relevant file paths to evidence chunk IDs."""

    relevant_paths = set(case.get("relevant", []))

    return {
        chunk["evidence_id"]
        for chunk in chunks
        if chunk["file_path"] in relevant_paths
    }


def _budget_select(
    ranked: list[dict],
) -> list[dict]:
    """Select ranked evidence within the configured context budget."""

    budget = select_with_budget(
        ranked,
        max_tokens=MAX_TOKENS,
    )

    selected_ids = set(
        budget["selected_ids"]
    )

    return [
        result
        for result in ranked
        if result["evidence_id"] in selected_ids
    ]


def run_baseline(
    chunks: list[dict],
    query: str,
) -> dict:
    """Run lexical retrieval without an evidence gate."""

    start = time.perf_counter()

    ranked = lexical_search(
        chunks,
        query,
        top_k=TOP_K,
    )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000.0

    selected = _budget_select(ranked)

    retrieved_ids = [
        result["evidence_id"]
        for result in ranked
    ]

    selected_ids = [
        result["evidence_id"]
        for result in selected
    ]

    # Baseline always reasons when evidence exists.
    if selected:
        conclusion = conclude(
            selected,
            query,
            client=None,
        )
    else:
        conclusion = None

    return {
        "mode": "lexical",
        "retrieved_ids": retrieved_ids,
        "selected_ids": selected_ids,
        "context_tokens": context_tokens(selected),
        "retrieval_latency_ms": round(
            latency_ms,
            3,
        ),
        "gate_outcome": None,
        "conclusion": conclusion,
        "llm_active": False,
    }


def run_current(
    chunks: list[dict],
    query: str,
) -> dict:
    """Run hybrid retrieval, budget, evidence gate and conditional reasoning."""

    start = time.perf_counter()

    ranked = hybrid_search(
        chunks,
        query,
        top_k=TOP_K,
    )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000.0

    selected = _budget_select(ranked)

    retrieved_ids = [
        result["evidence_id"]
        for result in ranked
    ]

    selected_ids = [
        result["evidence_id"]
        for result in selected
    ]

    client = (
        LLMClient()
        if is_llm_active()
        else None
    )

    llm_active = is_llm_active()
    llm_error = None

    try:
        result = analyze(
            selected,
            query,
            client=client,
        )

    except LLMError as exc:
        # If an API key exists but the LLM call fails, fall back to the
        # deterministic implementation so evaluation remains reproducible.
        result = analyze(
            selected,
            query,
            client=None,
        )

        llm_active = False
        llm_error = str(exc)

    return {
        "mode": "hybrid",
        "retrieved_ids": retrieved_ids,
        "selected_ids": selected_ids,
        "context_tokens": context_tokens(selected),
        "retrieval_latency_ms": round(
            latency_ms,
            3,
        ),
        "gate_outcome": result["outcome"],
        "reason": result["reason"],
        "missing": result["missing"],
        "conclusion": result["conclusion"],
        "llm_active": llm_active,
        "llm_error": llm_error,
    }


def _add_retrieval_metrics(
    record: dict,
    relevant: set[str],
) -> None:
    """Add deterministic retrieval metrics to an evaluation record."""

    retrieved = record["retrieved_ids"]

    record["relevant_ids"] = sorted(
        relevant
    )

    record["precision_at_k"] = precision_at_k(
        retrieved,
        relevant,
        TOP_K,
    )

    record["recall_at_k"] = recall_at_k(
        retrieved,
        relevant,
        TOP_K,
    )

    record["ndcg_at_k"] = ndcg_at_k(
        retrieved,
        relevant,
        TOP_K,
    )


def _add_gate_metrics(
    record: dict,
    ground_truth_sufficient: bool,
) -> None:
    """Add gate and abstention metrics."""

    outcome = record.get(
        "gate_outcome"
    )

    if outcome is None:
        # Baseline has no gate, so these metrics are undefined.
        record["gate_accuracy"] = None
        record["abstention_correct"] = None
        return

    record["gate_accuracy"] = gate_accuracy(
        outcome,
        ground_truth_sufficient,
    )

    record["abstention_correct"] = abstention_correct(
        outcome,
        ground_truth_sufficient,
    )


def _add_citation_metric(
    record: dict,
) -> None:
    """Evaluate citations while treating correct abstention as valid behavior.

    Important distinction:

        INSUFFICIENT + no conclusion
            = correct safe abstention

        SUFFICIENT + conclusion + valid citations
            = correct evidence-backed answer

        INSUFFICIENT + conclusion
            = unsafe behavior

        SUFFICIENT + missing/invalid citations
            = invalid evidence attribution
    """

    conclusion = record.get(
        "conclusion"
    )

    cited = (
        conclusion or {}
    ).get(
        "evidence_ids",
        [],
    )

    record["citation_valid"] = citation_valid(
        cited,
        set(record["selected_ids"]),
        conclusion_generated=conclusion is not None,
        gate_outcome=record.get("gate_outcome"),
    )


def evaluate_all(
    chunks: list[dict],
) -> dict:
    """Run every labeled query through both pipelines."""

    cases = load_queries()
    per_case = []

    for case in cases:
        query = case["query"]

        relevant = relevant_ids_for(
            chunks,
            case,
        )

        ground_truth = bool(
            case["sufficient"]
        )

        baseline = run_baseline(
            chunks,
            query,
        )

        current = run_current(
            chunks,
            query,
        )

        _add_retrieval_metrics(
            baseline,
            relevant,
        )

        _add_retrieval_metrics(
            current,
            relevant,
        )

        _add_gate_metrics(
            baseline,
            ground_truth,
        )

        _add_gate_metrics(
            current,
            ground_truth,
        )

        _add_citation_metric(
            baseline
        )

        _add_citation_metric(
            current
        )

        per_case.append(
            {
                "id": case["id"],
                "query": query,
                "ground_truth_sufficient": ground_truth,
                "baseline": baseline,
                "current": current,
            }
        )

    return {
        "top_k": TOP_K,
        "max_tokens": MAX_TOKENS,
        "cases": per_case,
    }


def summarize(
    report: dict,
) -> str:
    """Print a concise baseline versus current summary."""

    lines = []

    for case in report["cases"]:
        baseline = case["baseline"]
        current = case["current"]

        lines.append(
            f"[{case['id']}] "
            f"gt_sufficient="
            f"{case['ground_truth_sufficient']}\n"

            f"  baseline lex: "
            f"P@{TOP_K}="
            f"{baseline['precision_at_k']:.2f} "
            f"R@{TOP_K}="
            f"{baseline['recall_at_k']:.2f} "
            f"nDCG="
            f"{baseline['ndcg_at_k']:.2f} "
            f"tokens="
            f"{baseline['context_tokens']} "
            f"lat="
            f"{baseline['retrieval_latency_ms']:.1f}ms\n"

            f"  current hyb: "
            f"P@{TOP_K}="
            f"{current['precision_at_k']:.2f} "
            f"R@{TOP_K}="
            f"{current['recall_at_k']:.2f} "
            f"nDCG="
            f"{current['ndcg_at_k']:.2f} "
            f"tokens="
            f"{current['context_tokens']} "
            f"lat="
            f"{current['retrieval_latency_ms']:.1f}ms "
            f"gate="
            f"{current['gate_outcome']} "
            f"gate_acc="
            f"{current['gate_accuracy']} "
            f"abstention="
            f"{current['abstention_correct']} "
            f"citation_valid="
            f"{current['citation_valid']}\n"
        )

    return "\n".join(lines)


def write_report(
    report: dict,
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """Write the machine-readable evaluation report."""

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        results_dir
        / "evaluation_report.json"
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path


def main() -> None:
    """Execute the complete evaluation."""

    chunks = asyncio.run(
        load_chunks()
    )

    if not chunks:
        raise SystemExit(
            "No evidence chunks indexed. "
            "Run `python -m server.app.ingest` first."
        )

    report = evaluate_all(
        chunks
    )

    report["summary"] = {
        "chunk_count": len(chunks),
        "llm_active": is_llm_active(),
        "llm_dependent_metrics_available": False,
    }

    output_path = write_report(
        report
    )

    print(
        summarize(report)
    )

    print(
        f"Report written to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()