from __future__ import annotations

import json
from typing import Any, Dict, List

from .config import MEMORY_ROOT, settings

PROVIDERS_FILE = MEMORY_ROOT / "providers.json"


def _default_provider() -> Dict[str, Any]:
    return {
        "id": "openai",
        "label": "OpenAI-Compatible",
        "api_key": settings.api_key,
        "base_url": settings.base_url,
        "model": settings.model,
        "organization": settings.organization,
        "project": settings.project,
    }


def load_providers() -> Dict[str, Any]:
    if not PROVIDERS_FILE.exists():
        data = {
            "active_provider_id": "openai",
            "providers": [_default_provider()],
        }
        save_providers(data)
        return data
    try:
        data = json.loads(PROVIDERS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = {}
    providers = data.get("providers") or []
    if not providers:
        providers = [_default_provider()]
    active_id = data.get("active_provider_id") or providers[0].get("id", "openai")
    return {"active_provider_id": active_id, "providers": providers}


def save_providers(data: Dict[str, Any]) -> None:
    PROVIDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROVIDERS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_active_provider() -> Dict[str, Any]:
    data = load_providers()
    active_id = data.get("active_provider_id")
    for provider in data.get("providers", []):
        if provider.get("id") == active_id:
            return _normalize_provider(provider)
    providers = data.get("providers", [])
    return _normalize_provider(providers[0] if providers else _default_provider())


def upsert_provider(provider: Dict[str, Any], set_active: bool = True) -> Dict[str, Any]:
    data = load_providers()
    providers: List[Dict[str, Any]] = data.get("providers", [])
    normalized = _normalize_provider(provider)
    updated = False
    for idx, existing in enumerate(providers):
        if existing.get("id") == normalized.get("id"):
            providers[idx] = normalized
            updated = True
            break
    if not updated:
        providers.append(normalized)
    data["providers"] = providers
    if set_active:
        data["active_provider_id"] = normalized.get("id")
    save_providers(data)
    return normalized


def set_active_provider(provider_id: str) -> Dict[str, Any]:
    data = load_providers()
    data["active_provider_id"] = provider_id
    save_providers(data)
    return get_active_provider()


def _normalize_provider(provider: Dict[str, Any]) -> Dict[str, Any]:
    default = _default_provider()
    normalized = {**default, **provider}
    normalized["id"] = str(normalized.get("id") or normalized.get("label") or "provider").strip().lower()
    normalized["label"] = str(normalized.get("label") or normalized["id"]).strip()
    normalized["api_key"] = str(normalized.get("api_key") or "").strip()
    normalized["base_url"] = str(normalized.get("base_url") or "").strip()
    normalized["model"] = str(normalized.get("model") or "").strip()
    normalized["organization"] = str(normalized.get("organization") or "").strip()
    normalized["project"] = str(normalized.get("project") or "").strip()
    return normalized
