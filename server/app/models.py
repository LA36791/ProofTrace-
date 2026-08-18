"""SQLAlchemy ORM models for EvidenceRoom."""

from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from server.app.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Room(Base):
    __tablename__ = "rooms"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    issue: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="room", cascade="all, delete-orphan"
    )
    evidence_items: Mapped[list["Evidence"]] = relationship(
        back_populates="room", cascade="all, delete-orphan"
    )
    verdicts: Mapped[list["Verdict"]] = relationship(
        back_populates="room", cascade="all, delete-orphan"
    )
    conclusions: Mapped[list["Conclusion"]] = relationship(
        back_populates="room", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id"))
    author: Mapped[str] = mapped_column(String(255))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    room: Mapped[Room] = relationship(back_populates="messages")


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(primary_key=True)
    room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id"))
    chunk_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source: Mapped[str] = mapped_column(String(255))  # e.g. file path / URL / note
    content: Mapped[str] = mapped_column(Text)
    line_start: Mapped[int | None] = mapped_column(nullable=True)
    line_end: Mapped[int | None] = mapped_column(nullable=True)
    added_by: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    room: Mapped[Room] = relationship(back_populates="evidence_items")


class Verdict(Base):
    __tablename__ = "verdicts"

    id: Mapped[int] = mapped_column(primary_key=True)
    room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id"))
    outcome: Mapped[str] = mapped_column(String(20))  # SUFFICIENT | INSUFFICIENT
    reason: Mapped[str] = mapped_column(Text)
    missing_evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    room: Mapped[Room] = relationship(back_populates="verdicts")


class Conclusion(Base):
    __tablename__ = "conclusions"

    id: Mapped[int] = mapped_column(primary_key=True)
    room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id"))
    statement: Mapped[str] = mapped_column(Text)
    evidence_ids: Mapped[str] = mapped_column(Text)  # JSON array of evidence IDs
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    room: Mapped[Room] = relationship(back_populates="conclusions")
