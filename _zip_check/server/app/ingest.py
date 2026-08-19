"""Ingest the demo repository into SQLite as evidence chunks.

Run via: python -m server.app.ingest
"""

import asyncio
import hashlib
import re
from pathlib import Path

from sqlalchemy import select

from server.app.db import SessionLocal, init_db
from server.app.models import Evidence, Room

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT / "data" / "demo_repo"
DEMO_ROOM_TITLE = "Demo Investigation"

# Top-level definitions we treat as chunk boundaries.
_DEF_RE = re.compile(
    r"^(class\s+\w+|def\s+\w+|async\s+def\s+\w+|@\w)"  # includes decorators
)


def chunk_source(text: str, rel_path: str, max_lines: int = 60):
    """Split a source file into logical chunks by definition boundaries.

    Returns a list of dicts: {file_path, line_start, line_end, text}.
    """
    lines = text.splitlines()
    boundaries = [0]
    for i, line in enumerate(lines):
        if i > 0 and _DEF_RE.match(line):
            boundaries.append(i)
    boundaries.append(len(lines))

    chunks = []
    for start, end in zip(boundaries, boundaries[1:]):
        body = "\n".join(lines[start:end])
        if not body.strip():
            continue
        # Split oversized chunks into fixed-size buckets.
        body_lines = body.splitlines()
        if len(body_lines) > max_lines:
            for sub_start in range(0, len(body_lines), max_lines):
                sub = body_lines[sub_start : sub_start + max_lines]
                chunks.append(
                    {
                        "file_path": rel_path,
                        "line_start": start + sub_start + 1,
                        "line_end": start + sub_start + len(sub),
                        "text": "\n".join(sub),
                    }
                )
        else:
            chunks.append(
                {
                    "file_path": rel_path,
                    "line_start": start + 1,
                    "line_end": end,
                    "text": body,
                }
            )
    return chunks


def evidence_id(file_path: str, line_start: int, line_end: int) -> str:
    """Deterministic, stable evidence ID from file + line range."""
    raw = f"{file_path}:{line_start}:{line_end}"
    return "ev_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def collect_repo_files(root: Path) -> list[Path]:
    """Recursively list .py files in the demo repo (deterministic order)."""
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def build_chunks(root: Path = REPO_ROOT) -> list[dict]:
    """Build all chunks for the demo repository."""
    chunks = []
    for path in collect_repo_files(root):
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for c in chunk_source(text, rel):
            c["evidence_id"] = evidence_id(c["file_path"], c["line_start"], c["line_end"])
            chunks.append(c)
    return chunks


async def ingest(root: Path = REPO_ROOT) -> int:
    """Clear and reload evidence chunks for the demo room. Returns chunk count."""
    await init_db()
    async with SessionLocal() as db:
        # Ensure the demo room exists (Evidence requires a room_id).
        result = await db.execute(
            select(Room).where(Room.title == DEMO_ROOM_TITLE)
        )
        room = result.scalar_one_or_none()
        if room is None:
            room = Room(title=DEMO_ROOM_TITLE)
            db.add(room)
            await db.flush()

        # Remove prior ingested chunks for the demo room.
        existing = await db.execute(
            select(Evidence).where(Evidence.room_id == room.id)
        )
        for ev in existing.scalars():
            await db.delete(ev)

        chunks = build_chunks(root)
        for c in chunks:
            db.add(
                Evidence(
                    room_id=room.id,
                    chunk_id=c["evidence_id"],
                    source=c["file_path"],
                    content=c["text"],
                    line_start=c["line_start"],
                    line_end=c["line_end"],
                    added_by="ingest",
                )
            )
        await db.commit()
        return len(chunks)


async def load_chunks(root: Path = REPO_ROOT) -> list[dict]:
    """Load ingested chunks from the DB for retrieval."""
    await init_db()
    async with SessionLocal() as db:
        result = await db.execute(
            select(Room).where(Room.title == DEMO_ROOM_TITLE)
        )
        room = result.scalar_one_or_none()
        if room is None:
            return []
        evs = await db.execute(
            select(Evidence).where(Evidence.room_id == room.id)
        )
        return [
            {
                "evidence_id": ev.chunk_id,
                "file_path": ev.source,
                "line_start": ev.line_start,
                "line_end": ev.line_end,
                "text": ev.content,
            }
            for ev in evs.scalars()
        ]


def main() -> None:
    count = asyncio.run(ingest())
    print(f"Ingested {count} chunks into SQLite.")


if __name__ == "__main__":
    main()
