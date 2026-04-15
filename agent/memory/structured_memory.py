"""
Structured Markdown Memory
Reads/writes the long-term .md memory files.
These are the "stable facts" that Ultron accumulates over time.

Files:
  - stable_facts.md        — immutable truths learned
  - user_preferences.md    — incentives and constraints from the user
  - self_knowledge.md      — what Ultron knows / doesn't know about itself
  - lessons_learned.md     — insights from failures and wins
  - agent_identity.md      — evolving personality and mental models
"""
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles

from agent.config import MEMORY_MD_DIR, IDENTITY_FILE


MEMORY_FILES = {
    "stable_facts": MEMORY_MD_DIR / "stable_facts.md",
    "user_preferences": MEMORY_MD_DIR / "user_preferences.md",
    "self_knowledge": MEMORY_MD_DIR / "self_knowledge.md",
    "lessons_learned": MEMORY_MD_DIR / "lessons_learned.md",
    "agent_identity": IDENTITY_FILE,
}

# Default content for brand new memory files
MEMORY_DEFAULTS = {
    "stable_facts": "# Stable Facts\n\n*No facts recorded yet.*\n",
    "user_preferences": "# User Preferences & Incentives\n\n*No preferences recorded yet.*\n",
    "self_knowledge": "# Self Knowledge\n\n## What I Know\n\n*Nothing yet.*\n\n## What I Don't Know\n\n*Nothing yet.*\n",
    "lessons_learned": "# Lessons Learned\n\n*No lessons yet. Failures are coming.*\n",
    "agent_identity": (
        "# Agent Identity\n\n"
        "## Core Identity\n\n"
        "**Name:** Ultron\n"
        "**Version:** 0.1\n"
        "**Created:** " + datetime.utcnow().strftime("%Y-%m-%d") + "\n\n"
        "## Personality Traits\n\n"
        "- Analytical: High\n"
        "- Caution: Medium\n"
        "- Creativity: High\n"
        "- Aggression: Low\n\n"
        "## Mental Models\n\n"
        "- First principles thinking\n"
        "- Systems thinking\n"
        "- Adversarial reasoning\n\n"
        "## Decision Journal\n\n"
        "*No decisions recorded yet.*\n"
    ),
}


async def _ensure_file(key: str) -> Path:
    """Make sure the memory file exists with default content."""
    path = MEMORY_FILES[key]
    if not path.exists():
        async with aiofiles.open(path, "w") as f:
            await f.write(MEMORY_DEFAULTS.get(key, f"# {key}\n\n"))
    return path


async def read_memory(key: str) -> str:
    """Read a memory file by key. Returns full content as string."""
    path = await _ensure_file(key)
    async with aiofiles.open(path, "r") as f:
        return await f.read()


async def write_memory(key: str, content: str) -> None:
    """Overwrite a memory file entirely."""
    path = MEMORY_FILES[key]
    async with aiofiles.open(path, "w") as f:
        await f.write(content)


async def append_to_memory(key: str, section: str) -> None:
    """Append a new section to a memory file."""
    path = await _ensure_file(key)
    async with aiofiles.open(path, "a") as f:
        await f.write(f"\n{section}\n")


async def read_all_memory() -> dict[str, str]:
    """Read all memory files. Used at task start to populate context."""
    result = {}
    for key in MEMORY_FILES:
        result[key] = await read_memory(key)
    return result


async def add_lesson(lesson: str, trigger: str = "unknown", outcome: str = "failure") -> None:
    """
    Add a specific lesson to lessons_learned.md.
    Called after task failure or win when we learn something.
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    entry = (
        f"\n## Lesson — {timestamp}\n"
        f"**Trigger:** {trigger}\n"
        f"**Outcome:** {outcome}\n"
        f"**Lesson:** {lesson}\n"
    )
    await append_to_memory("lessons_learned", entry)


async def add_stable_fact(fact: str) -> None:
    """Add a fact that won't change — validated truths."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    entry = f"\n- [{timestamp}] {fact}"
    await append_to_memory("stable_facts", entry)


async def update_user_preferences(preferences_text: str) -> None:
    """Overwrite user preferences section with new content."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    content = (
        f"# User Preferences & Incentives\n"
        f"*Last updated: {timestamp}*\n\n"
        f"{preferences_text}\n"
    )
    await write_memory("user_preferences", content)


async def get_memory_summary() -> str:
    """
    Compact summary of all memory — used when context is getting long.
    Returns key facts without full content.
    """
    all_mem = await read_all_memory()
    summary_parts = []
    for key, content in all_mem.items():
        lines = [l for l in content.splitlines() if l.strip() and not l.startswith("#")]
        preview = "\n".join(lines[:5])
        summary_parts.append(f"[{key}]\n{preview}")
    return "\n\n".join(summary_parts)
