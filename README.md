# EvidenceRoom

A real-time collaborative AI evidence investigation application. Users investigate a software
issue together, collect repository evidence, and an AI evidence gate determines whether the
available evidence is sufficient to support a conclusion.

This is the backend MVP only (the React frontend is not built yet).

## Getting started

### Prerequisites

- Python 3.12+ (this repo was built against the included `.venv`)
- The dependencies in `requirements.txt` (they are already installed in `.venv`)

### Install dependencies

If you are not using the provided `.venv`:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Start the backend

From the repo root:

```bash
.venv/Scripts/python -m uvicorn server.app.main:app --reload
```

(On Linux/macOS: `source .venv/bin/activate && uvicorn server.app.main:app --reload`)

The API will be available at `http://127.0.0.1:8000`. Interactive docs at
`http://127.0.0.1:8000/docs`.

### Configuration

Copy `.env.example` to `.env` to override defaults (e.g. `DATABASE_URL`). The default
SQLite database is created automatically on startup at `./evidence.db`.

## Endpoints

| Method | Path              | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/health`         | Health check → `{"status": "ok"}`   |
| POST   | `/rooms`          | Create a room                         |
| GET    | `/rooms`          | List rooms                           |
| GET    | `/rooms/{id}`     | Get a room                           |
| WS     | `/ws/{room_id}`   | Connect to a room (sends confirmation)|

## Project structure

```
server/
  app/
    main.py       # FastAPI app, routes, WebSocket
    db.py         # SQLite async engine/session (SQLAlchemy + aiosqlite)
    models.py     # ORM models: Room, Message, Evidence, Verdict, Conclusion
    schemas.py    # Pydantic schemas for the API objects
```

## Next steps (not yet built)

- Hybrid retrieval (lexical + semantic) over repository evidence
- Reranking + context budget
- Evidence sufficiency gate (SUFFICIENT → conclude / INSUFFICIENT → abstain)
- LLM-backed evidence-backed conclusions with evidence IDs
- Real-time collaborative React frontend
- Retrieval/sufficiency/conclusion evaluation harness
