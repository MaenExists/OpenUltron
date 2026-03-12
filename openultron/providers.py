from __future__ import annotations

import json
from typing import Any, Dict, List

from openai import AsyncOpenAI

from .config import MEMORY_ROOT, settings

PROVIDERS_FILE = MEMORY_ROOT / "providers.json"

PROVIDER_PRESETS: List[Dict[str, Any]] = [
    {
        "id": "openai",
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "kind": "cloud",
        "requires_key": True,
        "docs": "https://platform.openai.com/docs/api-reference/models",
    },
    {
        "id": "openrouter",
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o",
        "kind": "router",
        "requires_key": True,
        "docs": "https://openrouter.ai/docs",
    },
    {
        "id": "together",
        "label": "Together AI",
        "base_url": "https://api.together.xyz/v1",
        "model": "openai/gpt-oss-20b",
        "kind": "cloud",
        "requires_key": True,
        "docs": "https://docs.together.ai/docs/openai-api-compatibility",
    },
    {
        "id": "perplexity-v1",
        "label": "Perplexity (Agent API v1)",
        "base_url": "https://api.perplexity.ai/v1",
        "model": "openai/gpt-5-mini",
        "kind": "cloud",
        "requires_key": True,
        "docs": "https://docs.perplexity.ai/docs/agent-api/openai-compatibility",
    },
    {
        "id": "perplexity-v2",
        "label": "Perplexity (Agent API v2)",
        "base_url": "https://api.perplexity.ai/v2",
        "model": "openai/gpt-5-mini",
        "kind": "cloud",
        "requires_key": True,
        "docs": "https://docs.perplexity.ai/docs/agentic-research/openai-compatibility",
    },
    {
        "id": "lmstudio",
        "label": "LM Studio (Local)",
        "base_url": "http://localhost:1234/v1",
        "model": "local-model",
        "kind": "local",
        "requires_key": False,
        "docs": "https://lmstudio.ai/docs/developer/openai-compat/",
    },
    {
        "id": "ollama",
        "label": "Ollama (Local)",
        "base_url": "http://localhost:11434/v1",
        "model": "gpt-oss:20b",
        "kind": "local",
        "requires_key": False,
        "docs": "https://docs.ollama.com/api/openai-compatibility",
    },
    {
        "id": "vllm",
        "label": "vLLM (Local)",
        "base_url": "http://localhost:8000/v1",
        "model": "NousResearch/Meta-Llama-3-8B-Instruct",
        "kind": "local",
        "requires_key": False,
        "docs": "https://cookbook.openai.com/articles/gpt-oss/run-vllm/",
    },
]


def _default_provider() -> Dict[str, Any]:
    return {
        "id": "openai",
        "label": "OpenAI-Compatible",
        "api_key": settings.api_key,
        "base_url": settings.base_url,
        "model": settings.model,
        "organization": settings.organization,
        "project": settings.project,
        "headers": {},
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


def build_client(provider: Dict[str, Any]) -> AsyncOpenAI:
    normalized = _normalize_provider(provider)
    api_key = _effective_api_key(normalized)
    if not api_key:
        raise RuntimeError("Missing provider API key.")
    client_args: Dict[str, Any] = {"api_key": api_key}
    base_url = normalized.get("base_url")
    if base_url:
        client_args["base_url"] = base_url
    organization = normalized.get("organization")
    if organization:
        client_args["organization"] = organization
    project = normalized.get("project")
    if project:
        client_args["project"] = project
    headers = normalized.get("headers") or {}
    if headers:
        client_args["default_headers"] = headers
    try:
        return AsyncOpenAI(**client_args)
    except TypeError:
        client_args.pop("default_headers", None)
        return AsyncOpenAI(**client_args)


def _effective_api_key(provider: Dict[str, Any]) -> str:
    api_key = str(provider.get("api_key") or "").strip()
    if api_key:
        return api_key
    base_url = str(provider.get("base_url") or "").lower()
    if "localhost" in base_url or "127.0.0.1" in base_url:
        return "EMPTY"
    return ""


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
    headers = normalized.get("headers") or {}
    if isinstance(headers, str):
        try:
            headers = json.loads(headers)
        except json.JSONDecodeError:
            headers = {}
    normalized["headers"] = headers if isinstance(headers, dict) else {}
    return normalized
