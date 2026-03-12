from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from .config import MEMORY_ROOT
from .utils import local_date, utc_now_iso, safe_join

EXPERIENCES_DIR = MEMORY_ROOT / "experiences"


def append_experience(entry: str) -> Path:
    date_stamp = local_date()
    file_path = EXPERIENCES_DIR / f"{date_stamp}.md"
    header = f"# Experiences {date_stamp}\n"
    if not file_path.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(header, encoding="utf-8")
    entry_block = f"\n\n## {utc_now_iso()}\n{entry.strip()}\n"
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(entry_block)
    return file_path


def latest_experience_excerpt(lines: int = 24) -> str:
    file_path = _latest_experience_file()
    if not file_path:
        return "No experiences recorded yet."
    content_lines = file_path.read_text(encoding="utf-8").splitlines()
    if not content_lines:
        return "No experiences recorded yet."
    return "\n".join(content_lines[-lines:])


def latest_experience_entries(count: int = 4) -> List[Dict[str, str]]:
    file_path = _latest_experience_file()
    if not file_path:
        return []
    sections = file_path.read_text(encoding="utf-8").split("\n## ")
    entries: List[Dict[str, str]] = []
    for section in sections[1:]:
        lines = section.splitlines()
        if not lines:
            continue
        timestamp = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        entries.append({"timestamp": timestamp, "body": body})
    return entries[-count:]


def list_memory_files() -> List[str]:
    files: List[str] = []
    if not MEMORY_ROOT.exists():
        return files
    for path in MEMORY_ROOT.rglob("*.md"):
        if path.name == "state.md":
            continue
        relative = path.relative_to(MEMORY_ROOT).as_posix()
        files.append(relative)
    return sorted(files, reverse=True)


def read_memory_file(relative_path: str) -> str:
    if not relative_path:
        return "Select a memory file to view it here."
    try:
        full_path = safe_join(MEMORY_ROOT, relative_path)
    except ValueError:
        return "Invalid memory path."
    if not full_path.exists() or full_path.suffix != ".md":
        return "Memory file not found."
    return full_path.read_text(encoding="utf-8")


def _latest_experience_file() -> Path | None:
    if not EXPERIENCES_DIR.exists():
        return None
    files = sorted(EXPERIENCES_DIR.glob("*.md"))
    return files[-1] if files else None
