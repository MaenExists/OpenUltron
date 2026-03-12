from __future__ import annotations

import json
from typing import Any, Dict

from .config import MEMORY_ROOT, settings

RUNTIME_FILE = MEMORY_ROOT / "runtime_settings.json"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "loop_interval_seconds": settings.loop_interval_seconds,
    "auto_execute_actions": settings.auto_execute_actions,
    "max_actions_per_loop": settings.max_actions_per_loop,
    "max_loops": settings.max_loops,
    "max_stalls": settings.max_stalls,
    "shell_mode": settings.shell_mode,
    "shell_timeout_seconds": settings.shell_timeout_seconds,
    "shell_allowlist": ",".join(settings.shell_allowlist),
}


def load_runtime_settings() -> Dict[str, Any]:
    if not RUNTIME_FILE.exists():
        save_runtime_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS.copy()
    try:
        data = json.loads(RUNTIME_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = {}
    return _normalize_settings({**DEFAULT_SETTINGS, **data})


def save_runtime_settings(data: Dict[str, Any]) -> None:
    RUNTIME_FILE.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_settings(data)
    RUNTIME_FILE.write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def update_runtime_settings(updates: Dict[str, Any]) -> Dict[str, Any]:
    current = load_runtime_settings()
    current.update(updates)
    normalized = _normalize_settings(current)
    save_runtime_settings(normalized)
    return normalized


def _normalize_settings(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    normalized["loop_interval_seconds"] = int(data.get("loop_interval_seconds", settings.loop_interval_seconds))
    normalized["auto_execute_actions"] = _as_bool(data.get("auto_execute_actions", settings.auto_execute_actions))
    normalized["max_actions_per_loop"] = int(data.get("max_actions_per_loop", settings.max_actions_per_loop))
    normalized["max_loops"] = int(data.get("max_loops", settings.max_loops))
    normalized["max_stalls"] = int(data.get("max_stalls", settings.max_stalls))
    normalized["shell_mode"] = str(data.get("shell_mode", settings.shell_mode)).lower()
    normalized["shell_timeout_seconds"] = int(data.get("shell_timeout_seconds", settings.shell_timeout_seconds))
    normalized["shell_allowlist"] = str(data.get("shell_allowlist", ",".join(settings.shell_allowlist)))
    return normalized


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "on"}
