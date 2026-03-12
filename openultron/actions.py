from __future__ import annotations

import asyncio
import json
import re
import shlex
import subprocess
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx
from bs4 import BeautifulSoup

from .config import MEMORY_ROOT, ROOT, settings
from .utils import safe_join, utc_now_iso

QUEUE_FILE = MEMORY_ROOT / "actions_queue.md"
ACTION_LOG_DIR = MEMORY_ROOT / "actions"

ALLOWED_ACTION_TYPES = {
    "shell",
    "write_file",
    "append_file",
    "write_memory",
    "append_memory",
    "search_web",
    "fetch_url",
}


@dataclass
class Action:
    id: str
    created_at: str
    status: str
    type: str
    title: str
    payload: Dict[str, Any]
    result: str = ""
    error: str = ""
    source: str = "loop"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        return cls(
            id=str(data.get("id", uuid.uuid4().hex)),
            created_at=str(data.get("created_at", utc_now_iso())),
            status=str(data.get("status", "proposed")),
            type=str(data.get("type", "")).strip(),
            title=str(data.get("title", "")).strip() or "Untitled action",
            payload=dict(data.get("payload", {})),
            result=str(data.get("result", "")),
            error=str(data.get("error", "")),
            source=str(data.get("source", "loop")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_queue() -> List[Action]:
    if not QUEUE_FILE.exists():
        save_queue([])
        return []
    content = QUEUE_FILE.read_text(encoding="utf-8")
    data = _read_json_block(content)
    actions: List[Action] = []
    for item in data:
        action = Action.from_dict(item)
        if action.type and action.type in ALLOWED_ACTION_TYPES:
            actions.append(action)
    return actions


def save_queue(actions: Iterable[Action]) -> None:
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = [action.to_dict() for action in actions]
    content = "# Action Queue\n"
    content += f"updated_at: {utc_now_iso()}\n\n"
    content += "```json\n"
    content += json.dumps(payload, indent=2)
    content += "\n```\n"
    QUEUE_FILE.write_text(content, encoding="utf-8")


def queue_actions(items: Iterable[Dict[str, Any]], source: str = "loop") -> List[Action]:
    actions = load_queue()
    new_actions: List[Action] = []
    for item in items:
        action = Action.from_dict({**item, "source": source})
        if action.type not in ALLOWED_ACTION_TYPES:
            continue
        new_actions.append(action)
    if not new_actions:
        return []
    actions.extend(new_actions)
    save_queue(actions)
    return new_actions


def update_action(action_id: str, **updates: Any) -> Optional[Action]:
    actions = load_queue()
    updated: Optional[Action] = None
    for idx, action in enumerate(actions):
        if action.id == action_id:
            data = action.to_dict()
            data.update(updates)
            updated = Action.from_dict(data)
            actions[idx] = updated
            break
    if updated:
        save_queue(actions)
    return updated


def approve_action(action_id: str) -> Optional[Action]:
    return update_action(action_id, status="approved", error="")


def reject_action(action_id: str) -> Optional[Action]:
    return update_action(action_id, status="rejected")


def approve_all(action_ids: Iterable[str]) -> List[Action]:
    updated: List[Action] = []
    for action_id in action_ids:
        action = approve_action(action_id)
        if action:
            updated.append(action)
    return updated


def next_approved_action() -> Optional[Action]:
    for action in load_queue():
        if action.status == "approved":
            return action
    return None


def _read_json_block(content: str) -> List[Dict[str, Any]]:
    match = re.search(r"```json\s*(.*?)```", content, re.S)
    if not match:
        return []
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return []


def _safe_join_root(relative_path: str) -> Path:
    candidate = (ROOT / relative_path).resolve()
    if candidate == ROOT or ROOT in candidate.parents:
        return candidate
    raise ValueError("Path escapes project root.")


async def run_next_approved() -> Optional[Action]:
    action = next_approved_action()
    if not action:
        return None
    update_action(action.id, status="running")
    result_action = await execute_action(action)
    save_queue(load_queue())
    return result_action


async def run_all_approved(limit: int = 5) -> List[Action]:
    results: List[Action] = []
    for _ in range(limit):
        action = await run_next_approved()
        if not action:
            break
        results.append(action)
    return results


async def execute_action(action: Action) -> Action:
    try:
        if action.type == "shell":
            result = await _execute_shell(action.payload)
            return _finalize_action(action, result=result)
        if action.type == "write_file":
            result = _write_file(action.payload)
            return _finalize_action(action, result=result)
        if action.type == "append_file":
            result = _append_file(action.payload)
            return _finalize_action(action, result=result)
        if action.type == "write_memory":
            result = _write_memory(action.payload)
            return _finalize_action(action, result=result)
        if action.type == "append_memory":
            result = _append_memory(action.payload)
            return _finalize_action(action, result=result)
        if action.type == "search_web":
            result = await _search_web(action.payload)
            return _finalize_action(action, result=result)
        if action.type == "fetch_url":
            result = await _fetch_url(action.payload)
            return _finalize_action(action, result=result)
        raise ValueError(f"Unsupported action type: {action.type}")
    except Exception as exc:
        return _finalize_action(action, error=str(exc))


def _finalize_action(action: Action, result: str = "", error: str = "") -> Action:
    status = "failed" if error else "done"
    updated = update_action(
        action.id,
        status=status,
        result=_trim_text(result),
        error=error,
    )
    if updated:
        _append_action_log(updated)
        return updated
    action.status = status
    action.result = _trim_text(result)
    action.error = error
    _append_action_log(action)
    return action


async def _execute_shell(payload: Dict[str, Any]) -> str:
    cmd = payload.get("cmd")
    if not cmd:
        raise ValueError("Missing cmd for shell action.")
    parts = _normalize_command(cmd)
    allowlist = settings.shell_allowlist
    if not parts or parts[0] not in allowlist:
        raise ValueError(f"Command '{parts[0] if parts else ''}' not allowed.")

    def _run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            parts,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

    completed = await asyncio.to_thread(_run)
    output = completed.stdout + completed.stderr
    return output.strip() or "(no output)"


def _normalize_command(cmd: Any) -> List[str]:
    if isinstance(cmd, list):
        return [str(part) for part in cmd]
    if isinstance(cmd, str):
        return shlex.split(cmd)
    raise ValueError("Invalid cmd format.")


def _write_file(payload: Dict[str, Any]) -> str:
    path = payload.get("path")
    content = payload.get("content", "")
    if not path:
        raise ValueError("Missing path for write_file.")
    target = _safe_join_root(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(content), encoding="utf-8")
    return f"Wrote {target.relative_to(ROOT)}"


def _append_file(payload: Dict[str, Any]) -> str:
    path = payload.get("path")
    content = payload.get("content", "")
    if not path:
        raise ValueError("Missing path for append_file.")
    target = _safe_join_root(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(str(content))
    return f"Appended {target.relative_to(ROOT)}"


def _write_memory(payload: Dict[str, Any]) -> str:
    path = payload.get("path")
    content = payload.get("content", "")
    if not path:
        raise ValueError("Missing path for write_memory.")
    target = safe_join(MEMORY_ROOT, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(content), encoding="utf-8")
    return f"Wrote memory {path}"


def _append_memory(payload: Dict[str, Any]) -> str:
    path = payload.get("path")
    content = payload.get("content", "")
    if not path:
        raise ValueError("Missing path for append_memory.")
    target = safe_join(MEMORY_ROOT, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(str(content))
    return f"Appended memory {path}"


async def _search_web(payload: Dict[str, Any]) -> str:
    query = str(payload.get("query", "")).strip()
    if not query:
        raise ValueError("Missing query for search_web.")
    max_results = int(payload.get("max_results", 5))
    url = "https://duckduckgo.com/html/"
    params = {"q": query}
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
    results: List[str] = []
    for result in soup.select(".result__body"):
        title = result.select_one(".result__a")
        snippet = result.select_one(".result__snippet")
        if not title:
            continue
        line = f"{title.get_text(strip=True)}"
        if snippet:
            line += f" — {snippet.get_text(strip=True)}"
        results.append(line)
        if len(results) >= max_results:
            break
    return "\n".join(results) or "No results."


async def _fetch_url(payload: Dict[str, Any]) -> str:
    url = str(payload.get("url", "")).strip()
    if not url:
        raise ValueError("Missing url for fetch_url.")
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(url)
        response.raise_for_status()
        text = response.text
    save_to = payload.get("save_to")
    if save_to:
        target = safe_join(MEMORY_ROOT, save_to)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return _trim_text(text)


def _append_action_log(action: Action) -> None:
    ACTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    date_stamp = utc_now_iso().split(" ")[0]
    log_file = ACTION_LOG_DIR / f"{date_stamp}.md"
    header = f"# Action Log {date_stamp}\n"
    if not log_file.exists():
        log_file.write_text(header, encoding="utf-8")
    entry = [
        f"\n\n## {utc_now_iso()}",
        f"Action: {action.title}",
        f"Type: {action.type}",
        f"Status: {action.status}",
    ]
    if action.result:
        entry.append("Result:")
        entry.append(action.result)
    if action.error:
        entry.append("Error:")
        entry.append(action.error)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(entry) + "\n")


def _trim_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated)"
