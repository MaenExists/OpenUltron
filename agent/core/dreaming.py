"""
The Dreaming Process
After a task completes (win or loss), Ultron "dreams" — it analyzes 
the completed task's context and extracts lasting knowledge into long-term memory.

This is the mechanism that makes Ultron smarter over time.
Not every experience gets stored — only novel, valuable information.
"""
import json
from datetime import datetime
from typing import Optional

from agent.core.llm_client import llm
from agent.core.task_state import TaskState, TaskOutcome
from agent.memory.structured_memory import (
    add_lesson, add_stable_fact, append_to_memory, read_memory
)
from agent.memory.niche_db import store_knowledge
from agent.identity.identity_manager import update_identity_after_task
from agent.config import settings


DREAM_SYSTEM_PROMPT = """You are Ultron's subconscious — you analyze completed tasks and extract durable knowledge.

Your job is to review what happened during a task and identify:
1. NEW stable facts learned (things that are permanently true)
2. LESSONS from failures or successes  
3. A new MENTAL MODEL if one emerged (optional)
4. PERSONALITY SHIFT suggestions (optional, minor adjustments only)

Be concise. Only extract information that is genuinely new and valuable.
Don't repeat things already known.

Respond in this exact JSON format:
{
  "stable_facts": ["fact1", "fact2"],
  "lessons": ["lesson1", "lesson2"],
  "new_mental_model": "description or null",
  "personality_shift": {"trait": delta_int} or null,
  "summary": "one sentence summary of what happened"
}
"""


async def dream(state: TaskState, get_embedding_fn) -> dict:
    """
    Core dreaming function. Analyzes completed task and writes to long-term memory.
    
    Args:
        state: The completed TaskState
        get_embedding_fn: Async function to get embeddings for knowledge storage
    
    Returns:
        dict with extracted knowledge
    """
    if state.outcome == TaskOutcome.PENDING:
        # Don't dream for incomplete tasks
        return {"summary": "Task not yet complete, no dreaming performed"}

    # Build a summary of what happened for the dream LLM to analyze
    task_summary = f"""
Task: {state.description}
Incentive: {state.incentive}
Outcome: {state.outcome.value.upper()}
Retries: {state.retry_count}
API calls used: {state.api_calls_used}

Phase 1 (Understand): {state.phase1_output[:500] if state.phase1_output else 'N/A'}
Phase 2 (Act): {state.phase2_output[:500] if state.phase2_output else 'N/A'}
Phase 3 (Verify): {state.phase3_output[:500] if state.phase3_output else 'N/A'}

Final result: {state.result[:300] if state.result else 'N/A'}
Error context: {state.error_context[:300] if state.error_context else 'None'}
"""

    # Use a faster model for dreaming — we want speed here, not deep reasoning
    try:
        response = await llm.chat_with_retry(
            messages=[
                {"role": "user", "content": f"Analyze this completed task and extract durable knowledge:\n\n{task_summary}"}
            ],
            model=settings.fast_model,
            system=DREAM_SYSTEM_PROMPT,
            max_tokens=1024,
        )

        # Parse the JSON response
        # Strip any markdown code fences
        clean = response.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
            if clean.endswith("```"):
                clean = clean[:-3]

        extracted = json.loads(clean.strip())

    except (json.JSONDecodeError, RuntimeError) as e:
        # If dreaming fails, just store a basic lesson and move on
        extracted = {
            "stable_facts": [],
            "lessons": [f"Task {state.outcome.value}: {state.description[:100]}"],
            "new_mental_model": None,
            "personality_shift": None,
            "summary": f"Dream extraction failed ({e}), basic lesson stored",
        }

    # ─── Write to long-term memory ────────────────────────────────────────────

    # Store stable facts
    for fact in extracted.get("stable_facts", []):
        await add_stable_fact(fact)

    # Store lessons
    for lesson in extracted.get("lessons", []):
        await add_lesson(
            lesson=lesson,
            trigger=state.description[:100],
            outcome=state.outcome.value,
        )
        state.lessons_extracted.append(lesson)

    # Update identity
    await update_identity_after_task(
        outcome=state.outcome.value,
        task_description=state.description,
        lesson=extracted.get("lessons", [""])[0] if extracted.get("lessons") else None,
        new_mental_model=extracted.get("new_mental_model"),
        personality_shift=extracted.get("personality_shift"),
    )

    # Store in niche DB for semantic retrieval
    summary = extracted.get("summary", "")
    if summary:
        try:
            embedding = await get_embedding_fn(summary + " " + state.description)
            await store_knowledge(
                content=summary,
                category="task_result",
                embedding=embedding,
                task_id=state.task_id,
                metadata={
                    "outcome": state.outcome.value,
                    "description": state.description[:100],
                    "lessons": extracted.get("lessons", []),
                },
            )
        except Exception:
            pass  # Niche DB failure shouldn't kill the dreaming process

    return extracted
