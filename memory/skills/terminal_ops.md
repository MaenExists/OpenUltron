# Terminal Ops Skill 🖥️

## Purpose

Use the terminal to inspect, verify, and change the local codebase safely.

## Pre‑Flight Checklist

- Confirm the working directory
- Decide whether allowlist or full mode is required
- Choose the smallest command that proves the point
- Set a timeout for any long‑running command

## Safe Execution Rules

- Prefer read‑only commands before making changes
- Use `rg` to locate files and patterns quickly
- Avoid destructive commands unless explicitly required
- Capture command output in memory if it affects decisions

## Common Commands

```bash
rg -n "TODO|FIXME" .
ls -la
pytest
```

## When to Use Full Shell Mode

- When you need pipes, redirects, or chained commands
- When a tool requires shell expansion
- Only if `OPENULTRON_SHELL_MODE=full` is enabled

## Post‑Run Validation

- Confirm files changed as expected
- Update memory with any new insight
- If errors occur, record them and propose a safer retry
