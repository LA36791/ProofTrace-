# ProofTrace

**Evidence-first AI debugging for software repositories.**

ProofTrace is a repository investigation system designed around one principle:

> An AI system should not produce a confident conclusion when the available repository evidence is insufficient to support it.

Instead of going directly from a user question to an AI-generated answer, ProofTrace retrieves repository evidence, ranks it, applies a context budget, evaluates whether the evidence is sufficient, and only then allows a conclusion to be generated. When the evidence is insufficient, the system explicitly abstains.

## Why I built it

AI coding systems are increasingly capable of producing plausible explanations for complex software problems. The difficult part is not only generating an answer, but knowing when the repository actually contains enough evidence to justify that answer.

I built ProofTrace to explore that problem from first principles.

The system compares a traditional lexical retrieval path with an evidence-first pipeline that combines retrieval, context selection, evidence sufficiency checking, conditional reasoning, and citation validation.

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
Top-K ranking
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
```

The important design decision is that **reasoning is conditional on evidence**.

## Key capabilities

### Evidence retrieval

Repository files are ingested into evidence chunks with stable evidence IDs.

ProofTrace supports:

* lexical retrieval
* hybrid retrieval
* top-K ranking
* context-budget selection

### Evidence sufficiency gate

The system explicitly distinguishes between:

* `SUFFICIENT` — the retrieved evidence supports proceeding toward a conclusion
* `INSUFFICIENT` — the available evidence does not justify a conclusion

This allows the system to abstain rather than manufacture an answer from incomplete evidence.

### Evidence-backed conclusions

When the evidence is sufficient, conclusions can reference the evidence IDs used to support them.

The system also validates that cited evidence belongs to the selected evidence set.

### Safe fallback behavior

The application can operate without an external LLM.

If an LLM is unavailable or an API call fails, the system falls back to the deterministic evidence-gating path instead of fabricating a conclusion.

### Context budgeting

Retrieved evidence is selected within a configurable token budget. This makes the tradeoff between evidence coverage and context size measurable rather than implicit.

### Evaluation harness

ProofTrace includes a deterministic evaluation harness comparing:

```text
BASELINE
query → lexical retrieval → context budget → reasoning

CURRENT
query → hybrid retrieval → context budget
      → evidence gate → conditional reasoning / abstention
```

The evaluation measures:

* Precision@K
* Recall@K
* nDCG@K
* context size
* retrieval latency
* gate accuracy
* correct abstention
* citation validity

## Current evaluation

The current evaluation contains **5 labeled investigation cases** over the demo repository.

The latest reproducible run produced:

```text
39 passed
```

and ingested:

```text
20 evidence chunks
```

One representative comparison:

| Metric                    |   Baseline |                             ProofTrace |
| ------------------------- | ---------: | -------------------------------------: |
| Discount case Precision@5 |       0.80 |                               **1.00** |
| Discount case Recall@5    |       0.50 |                               **0.62** |
| Discount case nDCG@5      |       0.87 |                               **1.00** |
| Discount case context     | 299 tokens |                         **234 tokens** |
| Evidence gate             |          — | Correct insufficient-evidence decision |
| No-match case             |          — |                     Correct abstention |

The evaluation is intentionally small and deterministic. These results are not presented as a production benchmark; they are used to validate the behavior and tradeoffs of the architecture.

## Frontend

The project includes a React/Vite frontend for interacting with the investigation system.

The frontend is responsible for presenting the investigation flow and making the evidence, decision, and conclusion understandable to the user.

Build it with:

```bash
cd client
npm install
npm run build
```

Run it locally with:

```bash
npm run dev
```

## Backend

The backend is implemented with FastAPI.

Start it locally from the repository root:

```bash
.venv/Scripts/python -m uvicorn server.app.main:app --reload
```

The API is available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## API

### Health

```text
GET /health
```

### Retrieval

```text
GET /retrieve
```

Supports lexical and hybrid retrieval.

Example:

```text
/retrieve?query=payment%20charge&mode=hybrid
```

### Investigation

```text
GET /analyze
```

Runs the complete evidence-first investigation pipeline.

Example:

```text
/analyze?query=payment%20charge
```

### Rooms

```text
POST /rooms
GET /rooms
GET /rooms/{id}
```

### WebSocket

```text
WS /ws/{room_id}
```

The room layer provides the foundation for real-time collaborative investigation.

## Repository structure

```text
ProofTrace/
│
├── server/
│   └── app/
│       ├── main.py
│       ├── db.py
│       ├── models.py
│       ├── schemas.py
│       ├── ingest.py
│       ├── retrieval/
│       ├── verifier/
│       └── evaluate/
│
├── client/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
│
├── data/
│   └── demo_repo/
│
├── eval/
│   ├── queries.json
│   └── results/
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Running the evaluation

From the repository root:

```bash
python -m server.app.ingest
python -m pytest -q
python -m server.app.evaluate.run
```

Expected test result for the current version:

```text
39 passed
```

The evaluation report is written to:

```text
eval/results/evaluation_report.json
```

## Design decisions

### 1. Evidence before reasoning

The central decision was to separate retrieval from reasoning.

This makes it possible to inspect what the system actually knows before allowing it to produce a conclusion.

### 2. Explicit abstention

I treated "I don't have enough evidence" as a valid system outcome rather than a failure.

This is especially important for debugging because an incorrect confident explanation can be more harmful than an incomplete answer.

### 3. Baseline versus current pipeline

Instead of evaluating ProofTrace in isolation, I kept a simpler lexical baseline.

This makes improvements and regressions measurable.

### 4. Deterministic evaluation

The core evaluation does not depend on an external LLM being available.

That makes the tests reproducible and allows retrieval and evidence-gating behavior to be evaluated independently from model availability.

### 5. Context budget as an explicit constraint

Repository intelligence can become expensive as context grows.

The context budget therefore becomes a measurable part of the retrieval pipeline rather than an implementation detail.

## What I would build next

If I continued developing ProofTrace, I would prioritize the following.

### 1. Repository-aware context graph

Move beyond independent chunks toward dependency-aware context.

For example, when a payment function is retrieved, the system should understand related callers, imported modules, data flow, and tests.

This would make evidence retrieval more useful for multi-file debugging.

### 2. Incremental repository indexing

Only changed files and affected dependencies should be re-indexed.

This would make ProofTrace practical for continuously changing repositories.

### 3. Investigation timeline

Expose the complete reasoning trail:

```text
Question
  ↓
Retrieved evidence
  ↓
Evidence selected / rejected
  ↓
Sufficiency decision
  ↓
Conclusion or abstention
```

This would make the system easier to audit and debug.

### 4. Deeper collaboration

The current room/WebSocket layer provides the foundation. I would next make collaboration meaningful by allowing multiple users or agents to contribute evidence, hypotheses, and observations to the same investigation.

## Limitations

ProofTrace is intentionally a focused prototype.

Current limitations include:

* the evaluation repository is small
* the evaluation set contains five labeled cases
* SQLite is used for the current persistence layer
* real-time collaboration is still an MVP
* semantic retrieval can operate through a deterministic fallback
* the evaluation is not intended to represent production-scale benchmark performance

These are deliberate scope decisions rather than claims of production readiness.

## What I learned

The main lesson from the project was that improving an AI system is not always about making the model generate more.

In an investigation workflow, knowing **when not to conclude** can be just as important as generating a good conclusion.

That led me to treat retrieval quality, context selection, evidence sufficiency, abstention, and citation validity as first-class parts of the product rather than treating them as implementation details around an LLM.

## Links

**GitHub:**
https://github.com/LA36791/ProofTrace-

**Live application:**
*Add deployed application URL here.*

**API documentation:**
*Add deployed `/docs` URL here.*

## License

This project was created as part of a technical product assignment.
