from __future__ import annotations

from pathlib import Path
from typing import Dict

from .config import MEMORY_ROOT
from .utils import utc_now_iso

STATE_FILE = MEMORY_ROOT / "state.md"

DEFAULT_STATE: Dict[str, str] = {
    "status": "paused",
    "brain_status": "idle",
    "loop_count": "0",
    "current_goal": "Define the next improvement target.",
    "last_cycle": "Never",
    "last_action": "None",
    "last_summary": "Awaiting first loop.",
    "last_error": "",
}

STATE_ORDER = [
    "status",
    "brain_status",
    "loop_count",
    "current_goal",
    "last_cycle",
    "last_action",
    "last_summary",
    "last_error",
]


def _parse_state(contents: str) -> Dict[str, str]:
    state = DEFAULT_STATE.copy()
    for line in contents.splitlines():
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        state[key.strip()] = value.strip().strip('"')
    return state


def read_state() -> Dict[str, str]:
    if not STATE_FILE.exists():
        write_state(DEFAULT_STATE)
        return DEFAULT_STATE.copy()
    return _parse_state(STATE_FILE.read_text(encoding="utf-8"))


def write_state(state: Dict[str, str]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# OpenUltron State", f"updated_at: {utc_now_iso()}"]
    for key in STATE_ORDER:
        value = state.get(key, DEFAULT_STATE.get(key, ""))
        lines.append(f"{key}: {value}")
    STATE_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_state(**updates: str) -> Dict[str, str]:
    state = read_state()
    state.update({k: str(v) for k, v in updates.items()})
    write_state(state)
    return state


def set_status(status: str) -> Dict[str, str]:
    brain_status = "observing" if status == "running" else "idle"
    return update_state(
        status=status,
        brain_status=brain_status,
        last_action=f"Status set to {status}",
    )


def set_goal(goal: str) -> Dict[str, str]:
    goal = goal.strip() or DEFAULT_STATE["current_goal"]
    return update_state(current_goal=goal, last_action="Goal updated")
