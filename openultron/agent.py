from __future__ import annotations

import asyncio
import hashlib

from .brain import Brain
from .config import settings
from .runtime import load_runtime_settings
from .memory import (
    append_experience,
    latest_experience_excerpt,
    recent_knowledge_excerpt,
    skills_index_excerpt,
    append_knowledge,
)
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
                runtime = load_runtime_settings()
                interval = int(runtime.get("loop_interval_seconds", settings.loop_interval_seconds))
                await asyncio.sleep(interval)
            else:
                await asyncio.sleep(1.0)

    async def loop_once(self) -> None:
        async with self._lock:
            state = read_state()
            runtime = load_runtime_settings()
            max_loops = int(runtime.get("max_loops", settings.max_loops))
            max_actions = int(runtime.get("max_actions_per_loop", settings.max_actions_per_loop))
            max_stalls = int(runtime.get("max_stalls", settings.max_stalls))
            auto_execute = bool(runtime.get("auto_execute_actions", settings.auto_execute_actions))

            if max_loops > 0:
                current_count = int(state.get("loop_count", "0"))
                if current_count >= max_loops:
                    update_state(
                        status="paused",
                        brain_status="idle",
                        last_error="Max loop count reached.",
                        last_action="Loop paused due to max loop limit.",
                    )
                    return

            try:
                update_state(brain_status="observing", last_action="Analyzing current context and memory...")
                context = _build_context()

                update_state(brain_status="thinking", last_action="Synthesizing neural pathways...")
                report = await self.brain.generate_loop_report(state, context)

                update_state(brain_status="acting", last_action=report.get("act", "Executing prime directive..."))
                entry = _format_entry(report)
                append_experience(entry)
                _record_learning(report)

                update_state(brain_status="evaluating", last_action=report.get("evaluate", "Evaluating outcome..."))

                update_state(brain_status="reflecting", last_action=report.get("reflect", "Reflecting on performance..."))

                update_state(brain_status="improving", last_action=report.get("improve", "Optimizing neural weights..."))

                actions = report.get("actions", [])
                queued = queue_actions(actions, source="loop")

                goal_complete = bool(report.get("goal_complete", False))
                stall_count, stalled = _update_stall_state(report, state, max_stalls)

                if auto_execute and queued and not goal_complete and not stalled:
                    approve_all([action.id for action in queued])
                    await run_all_approved(limit=max_actions)

                loop_count = int(state.get("loop_count", "0")) + 1
                status = "paused" if goal_complete or stalled else state.get("status", "running")
                last_error = ""
                if goal_complete:
                    last_error = "Goal marked complete. Loop paused."
                elif stalled:
                    last_error = "Stall detected. Loop paused for safety."

                update_state(
                    loop_count=str(loop_count),
                    last_cycle=utc_now_iso(),
                    last_action=report.get("act", "Cycle completed successfully."),
                    last_summary=report.get("summary", "Cycle completed successfully."),
                    last_summary_fingerprint=_fingerprint(report.get("summary", "")),
                    last_progress_at=utc_now_iso() if not stalled else state.get("last_progress_at", "Never"),
                    stall_count=str(stall_count),
                    last_goal_complete=str(goal_complete).lower(),
                    last_error=last_error,
                    brain_status="idle",
                    status=status,
                )
            except Exception as exc:
                append_experience(f"### Error\n{exc}")
                update_state(
                    status="paused",
                    last_error=str(exc),
                    last_action="Loop failed and paused.",
                    brain_status="error",
                )


def _build_context() -> str:
    parts = [
        "# Recent Experience",
        latest_experience_excerpt(),
        "# Knowledge Excerpt",
        recent_knowledge_excerpt(),
        "# Skills Index",
        skills_index_excerpt(),
    ]
    return "\n\n".join(parts)


def _record_learning(report: dict[str, object]) -> None:
    unknowns = _join_lines(report.get("unknowns", []))
    assumptions = _join_lines(report.get("assumptions", []))
    learnings = _join_lines(report.get("learnings", []))

    append_knowledge("knowledge/unknowns.md", unknowns, "Unknowns")
    append_knowledge("knowledge/assumptions.md", assumptions, "Assumptions")
    append_knowledge("knowledge/lessons.md", learnings, "Lessons")


def _join_lines(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _fingerprint(text: str) -> str:
    normalized = " ".join(text.lower().split())
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _update_stall_state(
    report: dict[str, object],
    state: dict[str, str],
    max_stalls: int,
) -> tuple[int, bool]:
    previous_fp = state.get("last_summary_fingerprint", "")
    current_fp = _fingerprint(str(report.get("summary", "")))
    no_actions = not report.get("actions")
    stall_count = int(state.get("stall_count", "0"))
    stalled = False
    if current_fp and current_fp == previous_fp and no_actions:
        stall_count += 1
        stalled = stall_count >= max_stalls
    else:
        stall_count = 0
    return stall_count, stalled


def _format_entry(report: dict[str, object]) -> str:
    sections = [
        ("Observe", report.get("observe", "")),
        ("Think", report.get("think", "")),
        ("Act", report.get("act", "")),
        ("Evaluate", report.get("evaluate", "")),
        ("Reflect", report.get("reflect", "")),
        ("Improve", report.get("improve", "")),
        ("Next Focus", report.get("next_focus", "")),
        ("Summary", report.get("summary", "")),
        ("Unknowns", _join_lines(report.get("unknowns", []))),
        ("Assumptions", _join_lines(report.get("assumptions", []))),
        ("Learnings", _join_lines(report.get("learnings", []))),
        ("Goal Progress", report.get("goal_progress", "")),
        ("Goal Complete", str(report.get("goal_complete", False))),
    ]
    actions = report.get("actions", [])
    if actions:
        action_lines = [
            f"- {action.get('title', 'Action')} ({action.get('type', 'unknown')})" for action in actions
        ]
        sections.append(("Proposed Actions", "\n".join(action_lines)))
    body_lines = [f"### {title}\n{content}" for title, content in sections if str(content).strip()]
    return "\n\n".join(body_lines)
