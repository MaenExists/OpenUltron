from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def local_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def safe_join(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if root == candidate or root in candidate.parents:
        return candidate
    raise ValueError("Path escapes memory root.")
