"""
OpenUltron Configuration
Centralizes all settings — env vars, model choices, paths, sandboxing limits.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


# ─── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
WORKSPACE_DIR = BASE_DIR / "workspace"
MEMORY_MD_DIR = BASE_DIR / "agent" / "memory" / "long_term_md"
NICHE_DB_PATH = BASE_DIR / "agent" / "memory" / "niche_db" / "memory.db"
CODE_CHANGES_DIR = WORKSPACE_DIR / "code_changes"
LOGS_DIR = WORKSPACE_DIR / "logs"
IDENTITY_FILE = MEMORY_MD_DIR / "agent_identity.md"

# Ensure dirs exist at import time
for d in [WORKSPACE_DIR, MEMORY_MD_DIR, NICHE_DB_PATH.parent, CODE_CHANGES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    # ─── OpenCode Zen API ─────────────────────────────────────────────────────
    opencode_api_key: str = Field(default="", alias="OPENCODE_API_KEY")
    opencode_api_base: str = "https://opencode.ai/zen/v1"

    # Default model — use gemini-3-flash for speed/cost, bump to claude-sonnet-4 for heavy tasks
    default_model: str = Field(
        default="opencode/gemini-3-flash",
        alias="ULTRON_MODEL"
    )
    reasoning_model: str = Field(
        default="opencode/claude-sonnet-4",
        alias="ULTRON_REASONING_MODEL"
    )
    fast_model: str = Field(
        default="opencode/gemini-3-flash",
        alias="ULTRON_FAST_MODEL"
    )

    # ─── Agent Identity ────────────────────────────────────────────────────────
    agent_name: str = Field(default="Ultron", alias="ULTRON_NAME")

    # ─── API Rate Limiting ─────────────────────────────────────────────────────
    max_tokens_per_request: int = 8192
    max_api_calls_per_task: int = 50          # safety rail
    api_timeout_seconds: int = 120

    # ─── Task / Loop Config ───────────────────────────────────────────────────
    max_retries_on_failure: int = 3
    task_timeout_seconds: int = 600           # 10 min per task max
    context_window_limit: int = 100_000       # chars before we compress

    # ─── Self-Modification ─────────────────────────────────────────────────────
    enable_self_modification: bool = Field(
        default=False,
        alias="ULTRON_SELF_MOD"
    )
    self_mod_requires_validation: bool = True

    # ─── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = Field(default=False, alias="DEBUG")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True


# Singleton — imported everywhere
settings = Settings()
