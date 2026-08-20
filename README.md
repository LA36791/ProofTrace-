# ProofTrace

**Evidence-first AI debugging for software repositories.**

ProofTrace is a repository investigation system built around one principle:

> An AI system should not produce a confident conclusion when the available repository evidence is insufficient to support it.

Instead of going directly from a question to an AI-generated answer, ProofTrace retrieves repository evidence, reranks it using corrective retrieval, applies a context budget, evaluates evidence sufficiency, and only then allows reasoning to proceed. When evidence is insufficient, the system abstains.

## Why I built it

AI coding systems can produce plausible explanations even when repository evidence is incomplete. The harder problem is determining whether the available evidence actually supports the conclusion.

ProofTrace explores this problem by separating:

- retrieval
- corrective ranking
- context selection
- evidence verification
- conditional reasoning
- citation validation
- safe abstention

The central design decision is simple:

**reasoning is conditional on evidence.**

## Core pipeline

```text
User question
      ↓
Repository ingestion
      ↓
Evidence chunks
      ↓
Lexical / Hybrid retrieval
      ↓
Corrective retrieval
      ↓
Top-K + source diversity
      ↓
Context-budget selection
      ↓
Evidence sufficiency gate
      ↓
 ┌───────────────────────┐
 │                       │
SUFFICIENT          INSUFFICIENT
 │                       │
 ↓                       ↓
Reason              Abstain safely
 │
 ↓
Evidence-backed conclusion
 │
 ↓
Evidence IDs / citations