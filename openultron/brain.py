from __future__ import annotations

import json
import re
from typing import Dict, Any

import httpx

from .config import settings


class SiliconFlowClient:
    def __init__(self) -> None:
        self.api_key = settings.api_key
        self.base_url = settings.base_url.rstrip("/")

    async def chat(self, model: str, messages: list[dict[str, str]]) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Missing SILICONFLOW_API_KEY.")
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 800,
        }
        async with httpx.AsyncClient(timeout=40) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()


class Brain:
    def __init__(self) -> None:
        self.client = SiliconFlowClient()
        self.model = settings.model

    async def generate_loop_report(self, state: Dict[str, str], context: str) -> Dict[str, Any]:
        system = (
            "You are OpenUltron's brain. Return JSON with keys: "
            "observe, think, act, evaluate, reflect, improve, next_focus, summary, actions. "
            "Values must be concise strings, max 80 words each. "
            "actions must be a JSON array of objects with: type, title, payload. "
            "Allowed action types: shell, write_file, append_file, write_memory, append_memory, search_web, fetch_url. "
            "When you lack knowledge, create search_web then fetch_url actions, then write_memory to store notes. "
            "If no actions are needed, return actions as an empty list."
        )
        user = (
            f"Current goal: {state.get('current_goal')}\n"
            f"Loop count: {state.get('loop_count')}\n"
            f"Last summary: {state.get('last_summary')}\n\n"
            f"Recent memory excerpt:\n{context}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            response = await self.client.chat(self.model, messages)
            content = response["choices"][0]["message"]["content"]
            parsed = _parse_json_block(content)
            return _coerce_report(parsed)
        except Exception as exc:
            return _fallback_report(str(exc), state)


def _parse_json_block(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise


def _coerce_report(data: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["observe", "think", "act", "evaluate", "reflect", "improve", "next_focus", "summary"]
    report: Dict[str, Any] = {}
    for key in keys:
        value = data.get(key, "")
        report[key] = str(value).strip() or "Not provided."
    actions = data.get("actions", [])
    if not isinstance(actions, list):
        actions = []
    report["actions"] = actions[:5]
    return report


def _fallback_report(error: str, state: Dict[str, str]) -> Dict[str, Any]:
    return {
        "observe": "No external model response available.",
        "think": f"Operating in offline mode. Error: {error}",
        "act": "Write a minimal loop entry and wait for API access.",
        "evaluate": "Offline output is limited but consistent.",
        "reflect": "Ensure SiliconFlow API key is configured.",
        "improve": "Configure the API key to enable learning loops.",
        "next_focus": state.get("current_goal", "Define a goal."),
        "summary": "Offline loop entry recorded.",
        "actions": [],
    }
