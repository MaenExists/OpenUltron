# OpenUltron 🤖

An experimental agentic system inspired by Ultron — a self-improving, incentive-driven artificial intelligence.

## Features
- **3-Phase Execution**: Understand → Act → Verify for all tasks.
- **Hybrid Memory**: Markdown-based long-term facts and semantic vector search via SQLite.
- **The Dreaming Process**: Post-task consolidation of short-term history into permanent memory.
- **Identity Evolution**: Personality and mental models that shift based on wins and losses.
- **Sandboxed Execution**: Safe file and command operations limited to `/workspace`.

## Quick Start

### 1. Requirements
- Python 3.12+
- `uv` for package management
- OpenCode Zen API Key (set as `OPENCODE_API_KEY`)

### 2. Setup
```bash
# Clone and enter the repo
git clone git@github.com:MaenExists/OpenUltron.git
cd OpenUltron

# Install dependencies and sync environment
uv sync
```

### 3. Run
Start the web dashboard:
```bash
uv run python3 -m agent.cli serve
```
Open your browser at `http://0.0.0.0:8000`.

Or run a task directly from the CLI:
```bash
uv run python3 -m agent.cli run "Write a Python script that calculates prime numbers up to 100 in workspace/primes.py"
```

## Project Structure
- `agent/`: Core logic, memory, identity, and tools.
- `web/`: FastAPI dashboard with HTMX and Tailwind CSS.
- `workspace/`: Isolated area where the agent operates.
- `docker/`: Deployment and sandboxing containers.

## Warning
OpenUltron is an experimental system. It has the capability to modify its own code if enabled. Run only in isolated environments.

---
*Created by Maen*
