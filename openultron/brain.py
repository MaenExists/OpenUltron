from __future__ import annotations

import json
import re
from typing import Dict, Any, List

from .config import settings
from .providers import get_active_provider, build_client
from .runtime import load_runtime_settings


class Brain:
    def __init__(self) -> None:
        self.model = settings.model

    async def generate_loop_report(self, state: Dict[str, str], context: str) -> Dict[str, Any]:
        system = (
            "You are OpenUltron's brain. Return a STRICT JSON object with keys: "
            "observe, think, act, evaluate, reflect, improve, next_focus, summary, "
            "unknowns, assumptions, learnings, goal_progress, goal_complete, actions. "
            "All values must be concise. unknowns/assumptions/learnings should be short lists or sentences. "
            "goal_complete must be a boolean. actions must be an array of objects with: type, title, payload. "
            "Allowed action types: shell, write_file, append_file, write_memory, append_memory, search_web, fetch_url. "
            "Be truthful: explicitly state what you do not know and what you assume. "
            "When you lack knowledge, propose search_web then fetch_url actions and add the unknowns. "
            "If no actions are needed, return actions as an empty list."
        )
        user = (
            f"Current goal: {state.get('current_goal')}\n"
            f"Loop count: {state.get('loop_count')}\n"
            f"Last summary: {state.get('last_summary')}\n"
            f"Last error: {state.get('last_error')}\n\n"
            f"Context and memory:\n{context}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            provider = get_active_provider()
            client = build_client(provider)
            model = provider.get("model") or settings.model
            try:
                response = await client.chat.responses.create(
                    model=model,
                    input=messages,
                    temperature=0.4,
                    max_output_tokens=800,
                    response_format={"type": "json_object"},
                )
            except Exception:
                response = await client.chat.responses.create(
                    model=model,
                    input=messages,
                    temperature=0.4,
                    max_output_tokens=800,
                )
            content = _extract_text(response)
            parsed = _parse_json_block(content)
            return _coerce_report(parsed)
        except Exception as exc:
            return _fallback_report(str(exc), state)


def _extract_text(response: Any) -> str:
    if hasattr(response, "output_text"):
        return str(response.output_text or "").strip()
    if hasattr(response, "output") and response.output:
        output = response.output[0]
        if hasattr(output, "content") and output.content:
            item = output.content[0]
            if hasattr(item, "text"):
                return str(item.text or "").strip()
    if hasattr(response, "choices") and response.choices:
        return str(response.choices[0].message.get("content", "")).strip()
    return ""


def _parse_json_block(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() in {"none", "n/a", "no"}:
        return []
    return [text]


def _coerce_report(data: Dict[str, Any]) -> Dict[str, Any]:
    string_keys = [
        "observe",
        "think",
        "act",
        "evaluate",
        "reflect",
        "improve",
        "next_focus",
        "summary",
        "goal_progress",
    ]
    report: Dict[str, Any] = {}
    for key in string_keys:
        value = data.get(key, "")
        report[key] = str(value).strip() or "Not provided."

    report["unknowns"] = _normalize_list(data.get("unknowns"))
    report["assumptions"] = _normalize_list(data.get("assumptions"))
    report["learnings"] = _normalize_list(data.get("learnings"))

    goal_complete = data.get("goal_complete", False)
    report["goal_complete"] = bool(goal_complete)

    actions = data.get("actions", [])
    if not isinstance(actions, list):
        actions = []
    runtime = load_runtime_settings()
    max_actions = int(runtime.get("max_actions_per_loop", settings.max_actions_per_loop))
    report["actions"] = actions[:max_actions]
    return report


def _fallback_report(error: str, state: Dict[str, str]) -> Dict[str, Any]:
    return {
        "observe": "No external model response available.",
        "think": f"Operating in offline mode. Error: {error}",
        "act": "Write a minimal loop entry and wait for API access.",
        "evaluate": "Offline output is limited but consistent.",
        "reflect": "Ensure a provider API key is configured.",
        "improve": "Configure the provider to enable learning loops.",
        "next_focus": state.get("current_goal", "Define a goal."),
        "summary": "Offline loop entry recorded.",
        "unknowns": ["Model response unavailable."],
        "assumptions": ["System is running without external LLM."],
        "learnings": [],
        "goal_progress": "Blocked on model access.",
        "goal_complete": False,
        "actions": [],
    }
