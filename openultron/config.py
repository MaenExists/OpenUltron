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
    model: str = os.getenv("SILICONFLOW_MODEL", "openai/gpt-oss-120b")
    api_key: str = os.getenv("SILICONFLOW_API_KEY", "")
    base_url: str = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.com/v1")
    auto_execute_actions: bool = os.getenv("OPENULTRON_AUTO_EXECUTE", "false").lower() == "true"
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
