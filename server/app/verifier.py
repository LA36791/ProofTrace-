"""AI Evidence Gate and evidence-backed reasoning.

Pipeline step run by the /analyze endpoint:
    query -> hybrid retrieval -> context budget -> gate -> conditional reasoning

Two operation modes:

- API mode (real): when ``OPENAI_API_KEY`` is set, a small httpx client calls an
  OpenAI-compatible chat-completions endpoint with structured JSON prompts.
- Fallback mode (deterministic): when no key is present, the gate and conclusion
  use explicit rules so the pipeline and tests run hermetically. The fallback
  is NOT an LLM; it exists so /analyze and the evaluation suite remain
  verifiable without external credentials.

Important design principle:

    Relevant repository code != sufficient evidence.

A repository may contain code explaining a possible execution path while still
lacking the runtime evidence needed to establish what actually happened in a
specific reported failure. The deterministic gate therefore distinguishes
between code-tracing questions and incident/failure questions.
"""

from __future__ import annotations

import json
import os
import re

import httpx


_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
_API_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://api.openai.com/v1",
).rstrip("/")
_API_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Minimum number of evidence chunks for ordinary deterministic
# repository-code questions.
MIN_EVIDENCE = 2


class LLMError(RuntimeError):
    """Raised when the LLM call or its structured response is invalid."""


def _system_prompt() -> str:
    return (
        "You are an evidence gate for a code investigation tool. You are given "
        "a query and a list of retrieved evidence chunks, each with an "
        "evidence_id. Decide whether the evidence is SUFFICIENT or INSUFFICIENT "
        "to support a conclusion. Do not treat the presence of relevant code "
        "alone as sufficient when the query describes a specific observed "
        "failure that requires runtime, request, state, or external-system "
        "evidence. Respond only with valid JSON of the form:\n"
        '{"outcome": "SUFFICIENT"|"INSUFFICIENT", '
        '"reason": "<short rationale>", '
        '"missing": ["<specific missing evidence>", ...]} '
        "For INSUFFICIENT, missing must list the specific evidence that is "
        "absent. For SUFFICIENT, missing must be an empty list."
    )


def _conclude_prompt(query: str, evidence: list[dict]) -> str:
    items = "\n".join(
        f"[{e['evidence_id']}] "
        f"({e['file_path']}:{e['line_start']}-{e['line_end']}) "
        f"{e['text']}"
        for e in evidence
    )

    return (
        f"Query: {query}\n\n"
        f"Retrieved evidence:\n{items}\n\n"
        "Write a concise conclusion supported ONLY by the cited evidence. "
        "Do not introduce facts that are not present in the evidence. "
        "Respond with valid JSON: "
        '{"statement": "<conclusion>", '
        '"evidence_ids": ["<evidence_id>", ...]}. '
        "Every evidence_id must be one of the retrieved IDs above."
    )


def _parse_json(text: str) -> dict:
    """Extract and parse the first JSON object from an LLM response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise LLMError("LLM response contained no JSON object")

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMError(f"LLM returned invalid JSON: {exc}") from exc


class LLMClient:
    """Thin OpenAI-compatible chat client built on httpx."""

    def __init__(
        self,
        api_key: str = _API_KEY,
        base_url: str = _API_BASE_URL,
        model: str = _API_MODEL,
    ) -> None:
        self._key = api_key
        self._base_url = base_url
        self._model = model
        self._client = httpx.Client(timeout=60)

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
    ) -> str:
        resp = self._client.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [
                    {
                        "role": "system",
                        "content": system,
                    },
                    {
                        "role": "user",
                        "content": user,
                    },
                ],
                "temperature": temperature,
            },
        )

        try:
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(
                f"LLM response had an unexpected structure: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Evidence gate
# ---------------------------------------------------------------------------


def _is_incident_query(query: str) -> bool:
    """Detect queries describing an observed failure or incident.

    Incident queries often require runtime or external-system evidence in
    addition to repository code.

    This deterministic classifier is intentionally small and transparent so
    the evaluation behavior can be inspected and tested without an LLM.
    """
    q = query.lower()

    incident_signals = (
        "customers report",
        "reported",
        "failing",
        "failure",
        "failed",
        "fails",
        "out-of-stock",
        "out of stock",
        "at the time of",
        "actual vs",
        "actual total",
        "not reflected",
        "not applied",
        "production issue",
        "specific order",
        "triggered",
    )

    return any(signal in q for signal in incident_signals)


def _fallback_missing(query: str) -> list[str]:
    """Return explicit evidence requirements for an insufficient query."""

    if _is_incident_query(query):
        return [
            (
                "A reproducible failing case with the relevant inputs "
                "and expected vs actual result"
            ),
            (
                "Runtime or external-system evidence showing what happened "
                "during the failure"
            ),
        ]

    return [
        "Direct repository evidence relevant to the query",
        (
            "Additional evidence connecting the retrieved code "
            "to the requested conclusion"
        ),
    ]


def gate(
    evidence: list[dict],
    query: str,
    client: LLMClient | None = None,
) -> dict:
    """Return the evidence-gate decision.

    Returns:

        {
            "outcome": "SUFFICIENT" | "INSUFFICIENT",
            "reason": "...",
            "missing": [...]
        }

    The deterministic fallback intentionally does NOT use evidence count as
    the only signal. Incident/failure queries require a higher evidence bar
    because repository code alone cannot prove what happened in a particular
    runtime event.
    """

    # ---------------------------------------------------------------
    # Rule 1: No retrieved evidence is always insufficient.
    # ---------------------------------------------------------------
    if not evidence:
        return {
            "outcome": "INSUFFICIENT",
            "reason": "No evidence was retrieved for this query.",
            "missing": [
                "No repository evidence was retrieved for this query.",
            ],
        }

    # ---------------------------------------------------------------
    # Deterministic fallback.
    #
    # This path is used when no LLM client is supplied.
    # ---------------------------------------------------------------
    if client is None:

        # Incident questions require runtime/request/state evidence.
        #
        # This prevents a common failure mode:
        #
        #     relevant code found
        #             ->
        #     falsely assuming the incident is explained
        #
        # The repository can explain what SHOULD happen without proving
        # what DID happen.
        if _is_incident_query(query):
            return {
                "outcome": "INSUFFICIENT",
                "reason": (
                    "The query describes an observed failure, but repository "
                    "evidence alone does not establish what happened in the "
                    "specific failing case."
                ),
                "missing": _fallback_missing(query),
            }

        # Ordinary repository-code questions still use the minimum evidence
        # guard.
        if len(evidence) < MIN_EVIDENCE:
            return {
                "outcome": "INSUFFICIENT",
                "reason": (
                    f"Only {len(evidence)} evidence chunk(s) were retrieved; "
                    f"at least {MIN_EVIDENCE} are required for this "
                    "deterministic fallback."
                ),
                "missing": _fallback_missing(query),
            }

        return {
            "outcome": "SUFFICIENT",
            "reason": (
                f"Retrieved {len(evidence)} evidence chunks and the query "
                "can be addressed from repository evidence."
            ),
            "missing": [],
        }

    # ---------------------------------------------------------------
    # Real LLM evidence gate.
    # ---------------------------------------------------------------
    user = (
        f"Query: {query}\n\n"
        "Retrieved evidence:\n"
        + "\n".join(
            f"[{e['evidence_id']}] {e['text']}"
            for e in evidence
        )
    )

    raw = client.chat(
        _system_prompt(),
        user,
        temperature=0.0,
    )

    data = _parse_json(raw)

    outcome = str(
        data.get("outcome", "")
    ).upper().strip()

    if outcome not in (
        "SUFFICIENT",
        "INSUFFICIENT",
    ):
        raise LLMError(
            f"LLM returned invalid outcome: {data.get('outcome')!r}"
        )

    missing = data.get("missing", [])

    if outcome == "INSUFFICIENT" and not isinstance(missing, list):
        raise LLMError(
            "INSUFFICIENT outcome must include a 'missing' list"
        )

    if outcome == "SUFFICIENT":
        # A sufficient decision must not claim missing evidence.
        missing = []

    return {
        "outcome": outcome,
        "reason": str(
            data.get("reason", "")
        ).strip(),
        "missing": missing if isinstance(missing, list) else [],
    }


# ---------------------------------------------------------------------------
# Evidence-backed reasoning
# ---------------------------------------------------------------------------


def _fallback_conclusion(
    evidence: list[dict],
    query: str,
) -> dict:
    """Create a deterministic conclusion using only retrieved evidence."""

    ids = [
        e["evidence_id"]
        for e in evidence
    ]

    files = sorted(
        {
            e["file_path"]
            for e in evidence
        }
    )

    statement = (
        f'Based on retrieved evidence, the issue described by "{query}" '
        f"is supported by code in: {', '.join(files)}."
    )

    return {
        "statement": statement,
        "evidence_ids": ids,
    }


def conclude(
    evidence: list[dict],
    query: str,
    client: LLMClient | None = None,
) -> dict:
    """Generate an evidence-backed conclusion.

    Every cited evidence_id must exist in the retrieved evidence. This creates
    a hard citation boundary between retrieved evidence and generated text.
    """

    if not evidence:
        raise LLMError(
            "Cannot generate a conclusion without evidence"
        )

    valid_ids = {
        e["evidence_id"]
        for e in evidence
    }

    # ---------------------------------------------------------------
    # Deterministic fallback.
    # ---------------------------------------------------------------
    if client is None:
        result = _fallback_conclusion(
            evidence,
            query,
        )

    # ---------------------------------------------------------------
    # Real LLM reasoning.
    # ---------------------------------------------------------------
    else:
        raw = client.chat(
            "You are an evidence-backed reasoning assistant. "
            "Use ONLY the supplied evidence.",
            _conclude_prompt(
                query,
                evidence,
            ),
            temperature=0.0,
        )

        result = _parse_json(raw)

    cited = result.get(
        "evidence_ids",
        [],
    )

    if not isinstance(cited, list) or not cited:
        raise LLMError(
            "Conclusion must cite at least one evidence_id"
        )

    invalid = [
        citation
        for citation in cited
        if citation not in valid_ids
    ]

    if invalid:
        raise LLMError(
            "Conclusion cited non-retrieved evidence_id(s): "
            f"{invalid}"
        )

    statement = result.get(
        "statement",
        "",
    )

    if not isinstance(statement, str) or not statement.strip():
        raise LLMError(
            "Conclusion statement is empty"
        )

    return {
        "statement": statement.strip(),
        "evidence_ids": list(cited),
    }


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------


def analyze(
    evidence: list[dict],
    query: str,
    client: LLMClient | None = None,
) -> dict:
    """Run gate and generate a conclusion only when evidence is sufficient."""

    verdict = gate(
        evidence,
        query,
        client=client,
    )

    result: dict = {
        "query": query,
        "outcome": verdict["outcome"],
        "reason": verdict["reason"],
        "missing": verdict["missing"],
        "conclusion": None,
    }

    # Critical safety boundary:
    #
    # INSUFFICIENT -> abstain
    # SUFFICIENT   -> reason
    #
    if verdict["outcome"] == "SUFFICIENT":
        result["conclusion"] = conclude(
            evidence,
            query,
            client=client,
        )

    return result


def is_llm_active() -> bool:
    """Return True when a real LLM API key is configured."""
    return bool(_API_KEY)