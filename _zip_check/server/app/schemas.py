"""Pydantic schemas for EvidenceRoom API objects."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---- Room ----
class RoomCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    issue: Optional[str] = None


class RoomRead(ORMModel):
    id: int
    title: str
    issue: Optional[str] = None
    created_at: datetime


# ---- Message ----
class MessageCreate(BaseModel):
    author: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)


class MessageRead(ORMModel):
    id: int
    room_id: int
    author: str
    content: str
    created_at: datetime


# ---- Evidence ----
class EvidenceCreate(BaseModel):
    source: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)
    added_by: str = Field(min_length=1, max_length=255)


class EvidenceRead(ORMModel):
    id: int
    room_id: int
    source: str
    content: str
    added_by: str
    created_at: datetime


# ---- Verdict ----
class VerdictRead(ORMModel):
    id: int
    room_id: int
    outcome: str
    reason: str
    missing_evidence: Optional[str] = None
    created_at: datetime


# ---- Conclusion ----
class ConclusionRead(ORMModel):
    id: int
    room_id: int
    statement: str
    evidence_ids: str
    created_at: datetime
