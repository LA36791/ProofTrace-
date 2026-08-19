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
    correct_retrieval,
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://prooftrace-frontend-live.onrender.com",
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

    elif mode == "hybrid_corrective":
        # Keep Hybrid RAG as the retrieval engine.
        # Corrective RAG only evaluates and improves its candidates.
        candidate_k = min(
            len(chunks),
            max(top_k * 2, 8),
        )

        hybrid_candidates = hybrid_search(
            chunks,
            query,
            top_k=candidate_k,
        )

        correction = correct_retrieval(
            hybrid_candidates,
            query,
            top_k=top_k,
        )

        results = correction["results"]

    elif mode == "lexical":
        results = lexical_search(
            chunks,
            query,
            top_k=top_k,
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "mode must be 'lexical', 'hybrid', "
                "or 'hybrid_corrective'"
            ),
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
    Run the complete evidence-grounded investigation pipeline.

    Query
        -> Hybrid RAG retrieval
        -> Corrective RAG quality control
        -> Context budget
        -> Evidence sufficiency gate
        -> Conditional LLM reasoning
        -> Evidence-backed conclusion or abstention

    Hybrid RAG remains the primary retrieval mechanism.
    Corrective RAG is an additional deterministic quality-control layer.
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
            "corrected_count": 0,
            "selected_ids": [],
            "retrieval_pipeline": [
                "hybrid",
                "corrective",
                "budget",
                "verifier",
            ],
        }

    # ------------------------------------------------------------------
    # 1. Hybrid RAG
    #
    # Retrieve a slightly larger candidate pool so Corrective RAG has
    # enough evidence to evaluate. This does NOT replace Hybrid RAG.
    # ------------------------------------------------------------------

    candidate_k = min(
        len(chunks),
        max(top_k * 2, 8),
    )

    hybrid_candidates = hybrid_search(
        chunks,
        query,
        top_k=candidate_k,
    )

    # ------------------------------------------------------------------
    # 2. Corrective RAG
    #
    # Deterministically remove weak candidates, improve query coverage,
    # deduplicate evidence and encourage useful file diversity.
    # ------------------------------------------------------------------

    correction = correct_retrieval(
        hybrid_candidates,
        query,
        top_k=top_k,
    )

    corrected = correction["results"]

    # ------------------------------------------------------------------
    # 3. Context budget
    # ------------------------------------------------------------------

    budget = select_with_budget(
        corrected,
        max_tokens=max_tokens,
    )

    selected_ids = set(budget["selected_ids"])

    selected = [
        result
        for result in corrected
        if result["evidence_id"] in selected_ids
    ]

    # ------------------------------------------------------------------
    # 4. Conditional LLM reasoning
    # ------------------------------------------------------------------

    client = LLMClient() if is_llm_active() else None

    try:
        result = analyze(
            selected,
            query,
            client=client,
        )

        result["llm_active"] = is_llm_active()

    except LLMError as exc:
        # Never fabricate a conclusion when the external LLM fails.
        result = analyze(
            selected,
            query,
            client=None,
        )

        result["llm_active"] = False
        result["llm_error"] = str(exc)

    # ------------------------------------------------------------------
    # 5. Retrieval/evidence trace metadata
    # ------------------------------------------------------------------

    result["retrieved_count"] = len(hybrid_candidates)
    result["corrected_count"] = len(corrected)
    result["selected_ids"] = budget["selected_ids"]

    result["retrieval_pipeline"] = [
        "hybrid",
        "corrective",
        "budget",
        "verifier",
    ]

    result["corrective_status"] = correction["status"]
    result["corrective_reason"] = correction["reason"]
    result["query_coverage"] = correction["query_coverage"]
    result["evidence_files"] = correction["files"]

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