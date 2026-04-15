"""
Sandbox Tools — safe operations Ultron can perform within /workspace.
These are the building blocks of the agent's action capability.

All file ops are scoped to WORKSPACE_DIR.
Bash ops use an allowlist of safe commands.
"""
import asyncio
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from agent.config import WORKSPACE_DIR, CODE_CHANGES_DIR, settings

# ─── Allowlist of safe shell commands ─────────────────────────────────────────
ALLOWED_COMMANDS = {
    "ls", "cat", "echo", "grep", "find", "pwd", "wc", "head", "tail",
    "mkdir", "touch", "cp", "mv", "rm", "python3", "python", "uv",
    "git", "curl", "wget", "pip", "pip3", "which", "env", "date",
    "sort", "uniq", "awk", "sed", "tr", "cut", "diff", "patch",
    "chmod", "stat", "du", "df", "ps", "kill", "sleep",
}


def _resolve_safe_path(path_str: str) -> Path:
    """
    Resolve a path and ensure it stays within WORKSPACE_DIR.
    Raises ValueError if path escapes the sandbox.
    """
    base = WORKSPACE_DIR.resolve()
    target = (base / path_str).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError(f"Path escape detected: {path_str} resolves outside workspace")
    return target


def _is_command_allowed(cmd: str) -> bool:
    """Check if the first token of a command is in the allowlist."""
    first_token = cmd.strip().split()[0] if cmd.strip() else ""
    # Handle env vars and path prefixes
    base_cmd = os.path.basename(first_token)
    return base_cmd in ALLOWED_COMMANDS


# ─── File Operations ──────────────────────────────────────────────────────────

async def tool_read_file(path: str) -> dict:
    """Read a file from the workspace. Returns content or error."""
    try:
        safe_path = _resolve_safe_path(path)
        if not safe_path.exists():
            return {"success": False, "error": f"File not found: {path}"}
        if not safe_path.is_file():
            return {"success": False, "error": f"Not a file: {path}"}

        content = safe_path.read_text(encoding="utf-8", errors="replace")
        # Limit to 50KB to avoid context bloat
        if len(content) > 50_000:
            content = content[:50_000] + "\n\n[... truncated at 50KB ...]"

        return {"success": True, "content": content, "path": str(safe_path)}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Read error: {e}"}


async def tool_write_file(path: str, content: str) -> dict:
    """Write content to a file in the workspace. Creates directories as needed."""
    try:
        safe_path = _resolve_safe_path(path)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")
        return {"success": True, "path": str(safe_path), "bytes_written": len(content)}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Write error: {e}"}


async def tool_list_files(path: str = ".") -> dict:
    """List files in a directory within the workspace."""
    try:
        safe_path = _resolve_safe_path(path)
        if not safe_path.exists():
            return {"success": False, "error": f"Directory not found: {path}"}

        entries = []
        for item in sorted(safe_path.iterdir()):
            entries.append({
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None,
            })
        return {"success": True, "entries": entries, "path": str(safe_path)}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"List error: {e}"}


async def tool_delete_file(path: str) -> dict:
    """Delete a file from the workspace."""
    try:
        safe_path = _resolve_safe_path(path)
        if not safe_path.exists():
            return {"success": False, "error": f"Not found: {path}"}
        if safe_path.is_dir():
            shutil.rmtree(safe_path)
        else:
            safe_path.unlink()
        return {"success": True, "deleted": str(safe_path)}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Delete error: {e}"}


# ─── Bash Execution ────────────────────────────────────────────────────────────

async def tool_bash(command: str, timeout: int = 30) -> dict:
    """
    Execute a shell command within the workspace sandbox.
    Only allows commands from the ALLOWED_COMMANDS allowlist.
    Working directory is always WORKSPACE_DIR.
    """
    if not _is_command_allowed(command):
        first = command.strip().split()[0] if command.strip() else "?"
        return {
            "success": False,
            "error": f"Command '{first}' is not in the allowlist. Allowed: {sorted(ALLOWED_COMMANDS)}",
        }

    try:
        result = await asyncio.wait_for(
            asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(WORKSPACE_DIR),
            ),
            timeout=timeout,
        )
        stdout, stderr = await result.communicate()
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": stdout.decode("utf-8", errors="replace")[:10_000],  # cap output
            "stderr": stderr.decode("utf-8", errors="replace")[:2_000],
        }
    except asyncio.TimeoutError:
        return {"success": False, "error": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": f"Execution error: {e}"}


# ─── Self-Modification (guarded) ───────────────────────────────────────────────

async def tool_propose_code_change(
    filename: str,
    description: str,
    new_content: str,
) -> dict:
    """
    Propose a code change to Ultron's own source.
    Writes to /workspace/code_changes/ for review, not applied directly.
    Requires settings.enable_self_modification = True.
    """
    if not settings.enable_self_modification:
        return {
            "success": False,
            "error": "Self-modification is disabled. Set ULTRON_SELF_MOD=true to enable.",
        }

    safe_filename = re.sub(r"[^\w\-.]", "_", filename)
    proposal_path = CODE_CHANGES_DIR / safe_filename

    import time
    proposal = (
        f"# Code Change Proposal\n"
        f"# Target file: {filename}\n"
        f"# Description: {description}\n"
        f"# Proposed at: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}\n"
        f"# Status: PENDING VALIDATION\n\n"
        f"{new_content}"
    )

    proposal_path.write_text(proposal, encoding="utf-8")
    return {
        "success": True,
        "proposal_path": str(proposal_path),
        "message": f"Change proposed for '{filename}'. Validation required before application.",
    }


# ─── Web Research (minimal, safe) ─────────────────────────────────────────────

async def tool_web_fetch(url: str) -> dict:
    """
    Fetch content from a URL. Limited to safe/common domains for research.
    Used by Ultron to research solutions during Phase 1 (Understand & Model).
    """
    import httpx

    # Basic URL sanity check — no local IPs or file:// 
    if not url.startswith(("http://", "https://")):
        return {"success": False, "error": "Only http/https URLs allowed"}

    blocked_patterns = ["localhost", "127.", "0.0.0.", "192.168.", "10.", "172."]
    for pat in blocked_patterns:
        if pat in url:
            return {"success": False, "error": f"Blocked URL pattern: {pat}"}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, follow_redirects=True, headers={"User-Agent": "OpenUltron/0.1 Research Agent"})
            content = resp.text[:20_000]  # cap at 20KB
            return {
                "success": True,
                "url": str(resp.url),
                "status_code": resp.status_code,
                "content": content,
            }
    except Exception as e:
        return {"success": False, "error": f"Fetch error: {e}"}


# ─── Tool Registry ─────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "list_files": tool_list_files,
    "delete_file": tool_delete_file,
    "bash": tool_bash,
    "propose_code_change": tool_propose_code_change,
    "web_fetch": tool_web_fetch,
}

# Tool schemas for LLM function calling
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace sandbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a workspace directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: '.')"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command in the workspace (allowlisted commands only)",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch content from a URL for research purposes",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch (http/https only)"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_code_change",
            "description": "Propose a modification to Ultron's own source code (requires self-modification enabled)",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Target source file name"},
                    "description": {"type": "string", "description": "What this change does"},
                    "new_content": {"type": "string", "description": "Full new content for the file"},
                },
                "required": ["filename", "description", "new_content"],
            },
        },
    },
]


async def execute_tool(name: str, arguments: dict) -> dict:
    """
    Execute a tool by name with given arguments.
    Central dispatch — called by the agent loop when LLM wants to use a tool.
    """
    if name not in TOOL_REGISTRY:
        return {"success": False, "error": f"Unknown tool: {name}. Available: {list(TOOL_REGISTRY.keys())}"}

    tool_fn = TOOL_REGISTRY[name]
    try:
        result = await tool_fn(**arguments)
        return result
    except TypeError as e:
        return {"success": False, "error": f"Tool argument error for '{name}': {e}"}
    except Exception as e:
        return {"success": False, "error": f"Tool '{name}' crashed: {e}"}
