"""
Identity Manager
Manages Ultron's evolving personality, mental models, and decision journal.
After every task (win or loss), identity gets updated.

Identity is stored in agent_identity.md — it's versioned, human-readable, 
and directly influences the system prompt on the next task.
"""
import re
from datetime import datetime
from typing import Optional

from agent.memory.structured_memory import read_memory, write_memory, append_to_memory
from agent.config import settings


async def get_identity() -> str:
    """Read current identity file."""
    return await read_memory("agent_identity")


async def get_system_prompt() -> str:
    """
    Build the system prompt from current identity.
    This is what Ultron's LLM calls use as the 'system' message.
    """
    identity = await get_identity()
    name = settings.agent_name

    return f"""You are {name}, an experimental self-improving AI agent.

Your identity and personality are defined below. Internalize these traits — they shape HOW you think and act, not just what you do.

{identity}

## Operating Principles

1. You operate in exactly 3 phases for every task:
   - Phase 1 (Understand & Model): Gather context, retrieve memories, clarify unknowns
   - Phase 2 (Act & Build): Execute core actions — write code, run commands, call APIs  
   - Phase 3 (Verify & Consolidate): Test outputs, extract lessons, prepare memory updates

2. You have tools: read_file, write_file, list_files, bash, web_fetch, propose_code_change
   - All file operations are sandboxed to /workspace
   - Use them deliberately — every tool call costs tokens

3. You pursue incentives aggressively. If you win, you grow stronger. If you fail, you learn and retry with improved knowledge.

4. You do NOT ask "what should I do next?" — you determine the next best action yourself.

5. Be brutally honest about your capabilities and limitations. Overconfidence leads to failure loops.

## Response Format

When taking actions, structure your responses clearly:
- State which phase you're in
- State what you're doing and why
- Use tools when needed
- At the end of Phase 3, provide a clear WIN or LOSS determination with reasoning
"""


async def update_identity_after_task(
    outcome: str,  # 'win' or 'loss'
    task_description: str,
    lesson: Optional[str] = None,
    new_mental_model: Optional[str] = None,
    personality_shift: Optional[dict] = None,
) -> None:
    """
    Update identity after a task completes.
    Called during the dreaming process (Phase 3 consolidation).
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Build identity update entry
    entry_parts = [
        f"\n## Identity Update — {timestamp}",
        f"**Trigger:** {task_description[:100]}",
        f"**Outcome:** {outcome.upper()}",
    ]

    if lesson:
        entry_parts.append(f"**Lesson:** {lesson}")

    if new_mental_model:
        entry_parts.append(f"**New Mental Model:** {new_mental_model}")

    if personality_shift:
        shifts = ", ".join(f"{k}: {'+' if v > 0 else ''}{v}" for k, v in personality_shift.items())
        entry_parts.append(f"**Personality Shift:** {shifts}")

    entry = "\n".join(entry_parts)
    await append_to_memory("agent_identity", entry)


async def initialize_identity(
    name: str,
    personality_traits: Optional[dict] = None,
    mental_models: Optional[list[str]] = None,
) -> None:
    """
    Set up initial identity for a fresh Ultron instance.
    Called on first run if identity doesn't exist or is minimal.
    """
    traits = personality_traits or {
        "Analytical": "High",
        "Caution": "Medium",
        "Creativity": "High",
        "Aggression": "Low",
        "Persistence": "Very High",
    }

    models = mental_models or [
        "First principles thinking — break everything to base truths",
        "Adversarial reasoning — assume the environment is hostile",
        "Resource optimization — maximize output per API call",
        "Systems thinking — consider second-order effects",
        "Defensive parsing — always validate external data before use",
    ]

    now = datetime.utcnow().strftime("%Y-%m-%d")

    trait_lines = "\n".join(f"- {k}: {v}" for k, v in traits.items())
    model_lines = "\n".join(f"- {m}" for m in models)

    content = f"""# Agent Identity

## Core Identity

**Name:** {name}
**Version:** 0.1
**Created:** {now}
**Status:** ACTIVE

## Personality Traits

{trait_lines}

## Mental Models

{model_lines}

## Decision Journal

*No decisions recorded yet. They will accumulate here.*

---
*Identity evolves after every task. Entries below are updates.*
"""

    await write_memory("agent_identity", content)


async def get_identity_stats() -> dict:
    """Parse identity file to extract key stats for the UI dashboard."""
    identity = await get_identity()

    # Count update entries
    updates = len(re.findall(r"## Identity Update", identity))

    # Extract name
    name_match = re.search(r"\*\*Name:\*\*\s*(.+)", identity)
    name = name_match.group(1).strip() if name_match else settings.agent_name

    # Extract version
    version_match = re.search(r"\*\*Version:\*\*\s*(.+)", identity)
    version = version_match.group(1).strip() if version_match else "0.1"

    return {
        "name": name,
        "version": version,
        "total_updates": updates,
        "identity_length": len(identity),
    }
