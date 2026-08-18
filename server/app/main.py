"""ProofTrace FastAPI application."""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.app.db import get_session, init_db
from server.app.ingest import load_chunks
from server.app.models import Room
from server.app.retrieval import (
    hybrid_search,
    lexical_search,
    select_with_budget,
)
from server.app.retrieval.semantic import is_real_semantic
from server.app.schemas import RoomCreate, RoomRead
from server.app.verifier import LLMClient, LLMError, analyze, is_llm_active


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="EvidenceRoom",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
# The React/Vite frontend runs on port 5173 while FastAPI runs on port 8000.
# Browser requests therefore require explicit CORS permission.
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


@app.get("/retrieve")
async def retrieve(
    query: str,
    top_k: int = 5,
    mode: str = "lexical",
    max_tokens: int | None = None,
):
    chunks = await load_chunks()

    if mode == "hybrid":
        results = hybrid_search(
            chunks,
            query,
            top_k=top_k,
        )
    elif mode == "lexical":
        results = lexical_search(
            chunks,
            query,
            top_k=top_k,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail="mode must be 'lexical' or 'hybrid'",
        )

    payload: dict = {
        "query": query,
        "top_k": top_k,
        "mode": mode,
        "semantic_active": is_real_semantic(),
        "results": results,
    }

    if max_tokens is not None:
        payload["context_budget"] = select_with_budget(
            results,
            max_tokens=max_tokens,
        )

    return payload


# ---------------------------------------------------------------------------
# Evidence analysis
# ---------------------------------------------------------------------------


@app.get("/analyze")
async def analyze_endpoint(
    query: str,
    top_k: int = 5,
    max_tokens: int = 800,
):
    """
    Run the complete investigation pipeline:

    query
        -> hybrid retrieval
        -> context budget
        -> evidence sufficiency gate
        -> conditional LLM reasoning
        -> evidence-backed conclusion or abstention
    """

    chunks = await load_chunks()

    if not chunks:
        return {
            "query": query,
            "outcome": "INSUFFICIENT",
            "reason": "No evidence is indexed. Run ingestion first.",
            "missing": ["No indexed repository evidence"],
            "conclusion": None,
            "llm_active": is_llm_active(),
            "retrieved_count": 0,
            "selected_ids": [],
        }

    # 1. Hybrid retrieval
    ranked = hybrid_search(
        chunks,
        query,
        top_k=top_k,
    )

    # 2. Context-budget selection
    budget = select_with_budget(
        ranked,
        max_tokens=max_tokens,
    )

    selected_ids = set(budget["selected_ids"])

    selected = [
        result
        for result in ranked
        if result["evidence_id"] in selected_ids
    ]

    # 3. Conditional LLM reasoning
    client = LLMClient() if is_llm_active() else None

    try:
        result = analyze(
            selected,
            query,
            client=client,
        )

        result["llm_active"] = is_llm_active()

    except LLMError as exc:
        # If the external LLM is unavailable or authentication fails,
        # safely fall back to the deterministic evidence gate.
        #
        # This is intentional: the system must never fabricate a conclusion
        # simply because the LLM is unavailable.
        result = analyze(
            selected,
            query,
            client=None,
        )

        result["llm_active"] = False
        result["llm_error"] = str(exc)

    # 4. Include retrieval metadata for the frontend/evaluation harness
    result["retrieved_count"] = len(ranked)
    result["selected_ids"] = budget["selected_ids"]

    return result


# ---------------------------------------------------------------------------
# Rooms
# ---------------------------------------------------------------------------


@app.post(
    "/rooms",
    response_model=RoomRead,
    status_code=201,
)
async def create_room(
    payload: RoomCreate,
    db: AsyncSession = Depends(get_session),
):
    room = Room(
        title=payload.title,
        issue=payload.issue,
    )

    db.add(room)

    await db.commit()
    await db.refresh(room)

    return room


@app.get(
    "/rooms",
    response_model=list[RoomRead],
)
async def list_rooms(
    db: AsyncSession = Depends(get_session),
):
    result = await db.execute(
        select(Room).order_by(Room.id)
    )

    return result.scalars().all()


@app.get(
    "/rooms/{room_id}",
    response_model=RoomRead,
)
async def get_room(
    room_id: int,
    db: AsyncSession = Depends(get_session),
):
    room = await db.get(
        Room,
        room_id,
    )

    if room is None:
        raise HTTPException(
            status_code=404,
            detail="Room not found",
        )

    return room


# ---------------------------------------------------------------------------
# Real-time collaboration
# ---------------------------------------------------------------------------


@app.websocket("/ws/{room_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    room_id: int,
):
    await websocket.accept()

    await websocket.send_json(
        {
            "type": "connection",
            "room_id": room_id,
            "message": "connected",
        }
    )

    try:
        while True:
            data = await websocket.receive_text()

            await websocket.send_json(
                {
                    "type": "echo",
                    "room_id": room_id,
                    "data": data,
                }
            )

    except WebSocketDisconnect:
        pass
