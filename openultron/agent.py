from __future__ import annotations

import asyncio

from .brain import Brain
from .config import settings
from .memory import append_experience, latest_experience_excerpt
from .actions import queue_actions, approve_all, run_all_approved
from .state import read_state, update_state
from .utils import utc_now_iso


class Agent:
    def __init__(self) -> None:
        self.brain = Brain()
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    async def start_background(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run_forever())

    async def _run_forever(self) -> None:
        while True:
            state = read_state()
            if state.get("status") == "running":
                await self.loop_once()
                await asyncio.sleep(settings.loop_interval_seconds)
            else:
                await asyncio.sleep(1.0)

    async def loop_once(self) -> None:
        async with self._lock:
            state = read_state()
            try:
                update_state(brain_status="observing", last_action="Analyzing current context and memory...")
                context = latest_experience_excerpt()
                
                update_state(brain_status="thinking", last_action="Synthesizing neural pathways...")
                report = await self.brain.generate_loop_report(state, context)
                
                update_state(brain_status="acting", last_action=report.get("act", "Executing prime directive..."))
                entry = _format_entry(report)
                append_experience(entry)
                
                update_state(brain_status="evaluating", last_action=report.get("evaluate", "Evaluating outcome..."))
                
                update_state(brain_status="reflecting", last_action=report.get("reflect", "Reflecting on performance..."))
                
                update_state(brain_status="improving", last_action=report.get("improve", "Optimizing neural weights..."))

                actions = report.get("actions", [])
                queued = queue_actions(actions, source="loop")
                if settings.auto_execute_actions and queued:
                    approve_all([action.id for action in queued])
                    await run_all_approved(limit=3)
                
                loop_count = int(state.get("loop_count", "0")) + 1
                update_state(
                    loop_count=str(loop_count),
                    last_cycle=utc_now_iso(),
                    last_action=report.get("act", "Cycle completed successfully."),
                    last_summary=report.get("summary", "Cycle completed successfully."),
                    last_error="",
                    brain_status="idle"
                )
            except Exception as exc:
                append_experience(f"### Error\n{exc}")
                update_state(
                    status="paused",
                    last_error=str(exc),
                    last_action="Loop failed and paused.",
                    brain_status="error"
                )


def _format_entry(report: dict[str, str]) -> str:
    sections = [
        ("Observe", report["observe"]),
        ("Think", report["think"]),
        ("Act", report["act"]),
        ("Evaluate", report["evaluate"]),
        ("Reflect", report["reflect"]),
        ("Improve", report["improve"]),
        ("Next Focus", report["next_focus"]),
        ("Summary", report["summary"]),
    ]
    actions = report.get("actions", [])
    if actions:
        action_lines = [f"- {action.get('title', 'Action')} ({action.get('type', 'unknown')})" for action in actions]
        sections.append(("Proposed Actions", "\n".join(action_lines)))
    body_lines = [f"### {title}\n{content}" for title, content in sections]
    return "\n\n".join(body_lines)
