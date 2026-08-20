"""AI Evidence Gate and evidence-backed reasoning.

Pipeline:
    query -> retrieval -> context budget -> evidence gate -> conclusion

Two modes are supported:

1. LLM mode:
   When OPENAI_API_KEY is configured, the verifier uses an
   OpenAI-compatible chat-completions endpoint.

2. Deterministic fallback mode:
   When the LLM is unavailable, the verifier still provides
   deterministic evidence-gated behavior so tests can run
   without external credentials.

Important principle:

    Relevant repository code != proof of a specific runtime incident.

Repository code can explain what SHOULD happen, but it cannot by
itself prove what DID happen in a particular production failure.
"""

from __future__ import annotations

import json
import os
import re

import httpx


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if _API_KEY in {"YOUR_REAL_API_KEY", "your_real_api_key"}:
    _API_KEY = ""

_API_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://api.openai.com/v1",
).rstrip("/")

_API_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini",
)

# Minimum evidence required for ordinary repository-code questions.
MIN_EVIDENCE = 2


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Raised when the LLM call or structured response is invalid."""


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------


def _system_prompt() -> str:
    return (
        "You are an evidence gate for a code investigation tool. "
        "You are given a query and retrieved evidence chunks, each with "
        "an evidence_id. Decide whether the evidence is SUFFICIENT or "
        "INSUFFICIENT to support a conclusion.\n\n"
        "Important rule: do not treat relevant repository code alone as "
        "sufficient when the query describes a specific observed failure "
        "that requires runtime, request, state, database, payment, or "
        "external-system evidence.\n\n"
        "Respond only with valid JSON in this form:\n"
        '{"outcome":"SUFFICIENT"|"INSUFFICIENT",'
        '"reason":"<short rationale>",'
        '"missing":["<specific missing evidence>", ...]}\n\n'
        "For INSUFFICIENT, missing must list the specific evidence that "
        "is absent. For SUFFICIENT, missing must be an empty list."
    )


def _conclude_prompt(
    query: str,
    evidence: list[dict],
) -> str:
    items = "\n".join(
        f"[{e['evidence_id']}] "
        f"({e['file_path']}:{e['line_start']}-{e['line_end']}) "
        f"{e['text']}"
        for e in evidence
    )

    return (
        f"Query: {query}\n\n"
        f"Retrieved evidence:\n{items}\n\n"
        "Write a concise conclusion supported ONLY by the supplied "
        "evidence. Do not introduce facts that are not present in the "
        "evidence.\n\n"
        "Respond with valid JSON:\n"
        '{"statement":"<conclusion>",'
        '"evidence_ids":["<evidence_id>", ...]}\n\n'
        "Every evidence_id must be one of the retrieved IDs."
    )


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _parse_json(text: str) -> dict:
    """Extract and parse the first JSON object from an LLM response."""

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL,
    )

    if not match:
        raise LLMError(
            "LLM response contained no JSON object"
        )

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"LLM returned invalid JSON: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


class LLMClient:
    """Thin OpenAI-compatible chat-completions client."""

    def __init__(
        self,
        api_key: str = _API_KEY,
        base_url: str = _API_BASE_URL,
        model: str = _API_MODEL,
    ) -> None:
        self._key = api_key
        self._base_url = base_url
        self._model = model

        self._client = httpx.Client(
            timeout=60,
        )

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
    ) -> str:
        response = self._client.post(
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
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(
                f"LLM request failed: {exc}"
            ) from exc

        try:
            data = response.json()

            return data["choices"][0]["message"]["content"]

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            raise LLMError(
                f"LLM response had an unexpected structure: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Incident detection
# ---------------------------------------------------------------------------


def _is_incident_query(query: str) -> bool:
    """Detect queries describing an observed failure or incident."""

    q = query.lower()

    incident_signals = (
        "customers report",
        "customer reported",
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
        "specific customer",
        "specific transaction",
        "triggered",
        "unexpected",
        "incorrect total",
        "wrong total",
        "charged the wrong",
        "charged incorrectly",
    )

    return any(
        signal in q
        for signal in incident_signals
    )


# ---------------------------------------------------------------------------
# Missing evidence
# ---------------------------------------------------------------------------


def _fallback_missing(
    query: str,
) -> list[str]:
    """Return explicit evidence requirements."""

    if _is_incident_query(query):
        return [
            (
                "A reproducible failing case with the relevant inputs "
                "and expected vs actual result"
            ),
            (
                "Runtime, request, state, database, or external-system "
                "evidence showing what happened during the failure"
            ),
        ]

    return [
        "Direct repository evidence relevant to the query",
        (
            "Additional evidence connecting the retrieved code "
            "to the requested conclusion"
        ),
    ]


# ---------------------------------------------------------------------------
# Evidence gate
# ---------------------------------------------------------------------------


def gate(
    evidence: list[dict],
    query: str,
    client: LLMClient | None = None,
) -> dict:
    """Return an evidence-gate decision."""

    # No evidence means we cannot conclude anything.
    if not evidence:
        return {
            "outcome": "INSUFFICIENT",
            "reason": (
                "No evidence was retrieved for this query."
            ),
            "missing": [
                "No repository evidence was retrieved for this query."
            ],
        }

    # -----------------------------------------------------------------------
    # Deterministic fallback
    # -----------------------------------------------------------------------

    if client is None:

        # Incident questions require stronger evidence.
        if _is_incident_query(query):
            return {
                "outcome": "INSUFFICIENT",
                "reason": (
                    "The query describes an observed failure, but "
                    "repository evidence alone does not establish what "
                    "happened in the specific failing case."
                ),
                "missing": _fallback_missing(query),
            }

        # Ordinary repository questions require at least two chunks.
        if len(evidence) < MIN_EVIDENCE:
            return {
                "outcome": "INSUFFICIENT",
                "reason": (
                    f"Only {len(evidence)} evidence chunk(s) were "
                    f"retrieved; at least {MIN_EVIDENCE} are required "
                    "for the deterministic fallback."
                ),
                "missing": _fallback_missing(query),
            }

        return {
            "outcome": "SUFFICIENT",
            "reason": (
                f"Retrieved {len(evidence)} evidence chunks and "
                "the query can be addressed from repository evidence."
            ),
            "missing": [],
        }

    # -----------------------------------------------------------------------
    # Real LLM gate
    # -----------------------------------------------------------------------

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
            f"LLM returned invalid outcome: "
            f"{data.get('outcome')!r}"
        )

    missing = data.get(
        "missing",
        [],
    )

    if (
        outcome == "INSUFFICIENT"
        and not isinstance(missing, list)
    ):
        raise LLMError(
            "INSUFFICIENT outcome must include a 'missing' list"
        )

    if outcome == "SUFFICIENT":
        missing = []

    return {
        "outcome": outcome,
        "reason": str(
            data.get("reason", "")
        ).strip(),
        "missing": (
            missing
            if isinstance(missing, list)
            else []
        ),
    }


# ---------------------------------------------------------------------------
# Deterministic evidence-backed conclusion
# ---------------------------------------------------------------------------


def _fallback_conclusion(
    evidence: list[dict],
    query: str,
) -> dict:
    """Create a deterministic conclusion from retrieved evidence.

    The fallback only states facts that are explicitly visible in the
    retrieved evidence.
    """

    ids = [
        e["evidence_id"]
        for e in evidence
    ]

    text = " ".join(
        str(e.get("text", "")).strip()
        for e in evidence
        if str(e.get("text", "")).strip()
    )

    query_lower = query.lower()

    parts: list[str] = []

    # -----------------------------------------------------------------------
    # Order / pricing flow
    # -----------------------------------------------------------------------

    if (
        "place_order" in query_lower
        or "final amount" in query_lower
        or "final total" in query_lower
        or "discount" in query_lower
        or "pricing" in query_lower
        or "tax" in query_lower
    ):

        if "subtotal = cart.subtotal()" in text:
            parts.append(
                "The order starts by calculating the cart subtotal."
            )

        if "apply_discount(subtotal" in text:
            parts.append(
                "The discount is applied to the subtotal."
            )

        if "compute_tax(discounted)" in text:
            parts.append(
                "Tax is calculated from the discounted amount."
            )

        if "total = discounted + tax" in text:
            parts.append(
                "The final total is the discounted amount plus tax."
            )

        if 'charge_card(card_token, totals["total"])' in text:
            parts.append(
                "The resulting total is passed to charge_card for payment."
            )

        if parts:
            return {
                "statement": " ".join(parts),
                "evidence_ids": ids,
            }

    # -----------------------------------------------------------------------
    # Payment flow
    # -----------------------------------------------------------------------

    if (
        "charge" in query_lower
        or "card" in query_lower
        or "payment" in query_lower
        or "declined" in query_lower
        or "invalid" in query_lower
    ):

        if "charge_card" in text:
            parts.append(
                "The retrieved payment code uses charge_card "
                "to process the card charge."
            )

        if "card_token" in text:
            parts.append(
                "The card token is supplied to the payment operation."
            )

        if parts:
            return {
                "statement": " ".join(parts),
                "evidence_ids": ids,
            }

    # -----------------------------------------------------------------------
    # Generic evidence-grounded fallback
    # -----------------------------------------------------------------------

    files = sorted(
        {
            str(e.get("file_path", ""))
            for e in evidence
            if e.get("file_path")
        }
    )

    if files:
        statement = (
            "Based on the retrieved repository evidence, "
            "the query is supported by the following code locations: "
            + ", ".join(files)
            + "."
        )
    else:
        statement = (
            "Based on the retrieved evidence, the query can be "
            "addressed from the supplied repository evidence."
        )

    return {
        "statement": statement,
        "evidence_ids": ids,
    }


# ---------------------------------------------------------------------------
# Conclusion
# ---------------------------------------------------------------------------


def conclude(
    evidence: list[dict],
    query: str,
    client: LLMClient | None = None,
) -> dict:
    """Generate an evidence-backed conclusion."""

    if not evidence:
        raise LLMError(
            "Cannot generate a conclusion without evidence"
        )

    valid_ids = {
        e["evidence_id"]
        for e in evidence
    }

    # -----------------------------------------------------------------------
    # Deterministic fallback
    # -----------------------------------------------------------------------

    if client is None:
        result = _fallback_conclusion(
            evidence,
            query,
        )

    # -----------------------------------------------------------------------
    # Real LLM reasoning
    # -----------------------------------------------------------------------

    else:
        raw = client.chat(
            (
                "You are an evidence-backed reasoning assistant. "
                "Use ONLY the supplied evidence."
            ),
            _conclude_prompt(
                query,
                evidence,
            ),
            temperature=0.0,
        )

        result = _parse_json(raw)

    # -----------------------------------------------------------------------
    # Validate citations
    # -----------------------------------------------------------------------

    cited = result.get(
        "evidence_ids",
        [],
    )

    if (
        not isinstance(cited, list)
        or not cited
    ):
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

    if (
        not isinstance(statement, str)
        or not statement.strip()
    ):
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


# ---------------------------------------------------------------------------
# LLM status
# ---------------------------------------------------------------------------


def is_llm_active() -> bool:
    """Return True when a real LLM API key is configured."""

    return bool(_API_KEY)
