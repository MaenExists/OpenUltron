"""
Niche Database — SQLite + sqlite-vec
Stores task-specific knowledge with vector embeddings for semantic search.
Ultron uses this to retrieve relevant past experiences when starting a new task.
"""
import asyncio
import json
import struct
import time
from datetime import datetime
from typing import Optional

import aiosqlite
import sqlite_vec

from agent.config import NICHE_DB_PATH


# Embedding dimensions (we use 64-dim pseudo-embeddings or real ones)
EMBED_DIM = 64


async def _get_db() -> aiosqlite.Connection:
    """Get a database connection with sqlite-vec loaded."""
    db = await aiosqlite.connect(str(NICHE_DB_PATH))
    # Load sqlite-vec extension for vector operations
    await db.enable_load_extension(True)
    try:
        await db.execute("SELECT load_extension(?)", (sqlite_vec.loadable_path(),))
    except Exception:
        # sqlite-vec may not support extension loading in all environments
        pass
    await db.enable_load_extension(False)
    return db


async def init_db() -> None:
    """
    Initialize the database schema.
    Creates tables for task memories and their embeddings.
    """
    db = await _get_db()
    try:
        # Main knowledge table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                category TEXT,          -- 'task_result', 'research', 'pattern', 'failure'
                content TEXT NOT NULL,
                metadata TEXT,          -- JSON blob for extra info
                embedding BLOB,         -- packed float32 vector
                created_at REAL,        -- unix timestamp
                access_count INTEGER DEFAULT 0,
                last_accessed REAL
            )
        """)

        # Task history table — tracks every task Ultron attempted
        await db.execute("""
            CREATE TABLE IF NOT EXISTS task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                incentive TEXT,
                outcome TEXT,           -- 'win', 'loss', 'timeout', 'in_progress'
                phase_reached INTEGER DEFAULT 1,
                error_context TEXT,
                lessons TEXT,
                started_at REAL,
                completed_at REAL,
                api_calls_used INTEGER DEFAULT 0
            )
        """)

        # Index for fast lookups
        await db.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge(category)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_task_history_outcome ON task_history(outcome)")

        await db.commit()
    finally:
        await db.close()


def _pack_embedding(embedding: list[float]) -> bytes:
    """Pack a float list into binary for sqlite storage."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _unpack_embedding(data: bytes) -> list[float]:
    """Unpack binary embedding back to float list."""
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


async def store_knowledge(
    content: str,
    category: str,
    embedding: list[float],
    task_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> int:
    """Store a piece of knowledge with its embedding."""
    db = await _get_db()
    try:
        now = time.time()
        emb_bytes = _pack_embedding(embedding[:EMBED_DIM] if len(embedding) >= EMBED_DIM else embedding + [0.0] * (EMBED_DIM - len(embedding)))
        cursor = await db.execute(
            """
            INSERT INTO knowledge (task_id, category, content, metadata, embedding, created_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                category,
                content,
                json.dumps(metadata or {}),
                emb_bytes,
                now,
                now,
            ),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def search_knowledge(
    query_embedding: list[float],
    category: Optional[str] = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Semantic search over stored knowledge.
    Returns top_k most similar entries by cosine similarity.
    """
    db = await _get_db()
    try:
        if category:
            rows = await db.execute_fetchall(
                "SELECT id, task_id, category, content, metadata, embedding, created_at FROM knowledge WHERE category = ?",
                (category,),
            )
        else:
            rows = await db.execute_fetchall(
                "SELECT id, task_id, category, content, metadata, embedding, created_at FROM knowledge",
            )

        # Compute cosine similarity for each row
        scored = []
        for row in rows:
            row_id, task_id, cat, content, metadata, emb_bytes, created_at = row
            if emb_bytes:
                emb = _unpack_embedding(emb_bytes)
                sim = _cosine_similarity(query_embedding[:EMBED_DIM], emb[:EMBED_DIM])
            else:
                sim = 0.0
            scored.append({
                "id": row_id,
                "task_id": task_id,
                "category": cat,
                "content": content,
                "metadata": json.loads(metadata or "{}"),
                "similarity": sim,
                "created_at": created_at,
            })

        # Sort by similarity, return top_k
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        results = scored[:top_k]

        # Update access counts for retrieved items
        if results:
            ids = [r["id"] for r in results]
            now = time.time()
            for rid in ids:
                await db.execute(
                    "UPDATE knowledge SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                    (now, rid),
                )
            await db.commit()

        return results
    finally:
        await db.close()


async def record_task(
    task_id: str,
    description: str,
    incentive: str = "",
    outcome: str = "in_progress",
) -> None:
    """Record a task starting or completing."""
    db = await _get_db()
    try:
        now = time.time()
        await db.execute(
            """
            INSERT OR REPLACE INTO task_history 
            (task_id, description, incentive, outcome, started_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (task_id, description, incentive, outcome, now),
        )
        await db.commit()
    finally:
        await db.close()


async def update_task_outcome(
    task_id: str,
    outcome: str,
    phase_reached: int = 3,
    error_context: str = "",
    lessons: str = "",
    api_calls_used: int = 0,
) -> None:
    """Update a task's final outcome."""
    db = await _get_db()
    try:
        now = time.time()
        await db.execute(
            """
            UPDATE task_history SET
                outcome = ?,
                phase_reached = ?,
                error_context = ?,
                lessons = ?,
                completed_at = ?,
                api_calls_used = ?
            WHERE task_id = ?
            """,
            (outcome, phase_reached, error_context, lessons, now, api_calls_used, task_id),
        )
        await db.commit()
    finally:
        await db.close()


async def get_task_history(limit: int = 20) -> list[dict]:
    """Retrieve recent task history for display in the UI."""
    db = await _get_db()
    try:
        rows = await db.execute_fetchall(
            """
            SELECT task_id, description, incentive, outcome, phase_reached,
                   error_context, lessons, started_at, completed_at, api_calls_used
            FROM task_history 
            ORDER BY started_at DESC 
            LIMIT ?
            """,
            (limit,),
        )
        return [
            {
                "task_id": r[0],
                "description": r[1],
                "incentive": r[2],
                "outcome": r[3],
                "phase_reached": r[4],
                "error_context": r[5],
                "lessons": r[6],
                "started_at": r[7],
                "completed_at": r[8],
                "api_calls_used": r[9],
            }
            for r in rows
        ]
    finally:
        await db.close()


async def get_win_loss_stats() -> dict:
    """Quick stats for the UI dashboard."""
    db = await _get_db()
    try:
        rows = await db.execute_fetchall(
            "SELECT outcome, COUNT(*) FROM task_history GROUP BY outcome"
        )
        stats = {r[0]: r[1] for r in rows}
        return {
            "wins": stats.get("win", 0),
            "losses": stats.get("loss", 0),
            "timeouts": stats.get("timeout", 0),
            "in_progress": stats.get("in_progress", 0),
            "total": sum(stats.values()),
        }
    finally:
        await db.close()
