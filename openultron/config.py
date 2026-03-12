from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
MEMORY_ROOT = ROOT / "memory"
TEMPLATES_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"


@dataclass(frozen=True)
class Settings:
    loop_interval_seconds: int = int(os.getenv("OPENULTRON_LOOP_INTERVAL", "12"))
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    organization: str = os.getenv("OPENAI_ORG", "")
    project: str = os.getenv("OPENAI_PROJECT", "")
    auto_execute_actions: bool = os.getenv("OPENULTRON_AUTO_EXECUTE", "false").lower() == "true"
    max_actions_per_loop: int = int(os.getenv("OPENULTRON_MAX_ACTIONS_PER_LOOP", "5"))
    max_loops: int = int(os.getenv("OPENULTRON_MAX_LOOPS", "0"))
    max_stalls: int = int(os.getenv("OPENULTRON_MAX_STALLS", "3"))
    shell_mode: str = os.getenv("OPENULTRON_SHELL_MODE", "allowlist").lower()
    shell_timeout_seconds: int = int(os.getenv("OPENULTRON_SHELL_TIMEOUT", "120"))
    shell_allowlist: list[str] = field(
        default_factory=lambda: [
            item.strip()
            for item in os.getenv(
                "OPENULTRON_SHELL_ALLOWLIST",
                "ls,rg,cat,sed,python,python3,pip,git,uvicorn,pytest",
            ).split(",")
            if item.strip()
        ]
    )


settings = Settings()
