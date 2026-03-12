# OpenUltron 🤖⚡

A persistent, filesystem-native agent that **observes → thinks → acts → learns** and builds its own memory over time.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![HTMX](https://img.shields.io/badge/HTMX-UI-0F172A)
![License](https://img.shields.io/badge/License-MIT-22C55E)

## Why OpenUltron ✨

OpenUltron is built for **long‑running, self‑improving agent workflows**. It stores memory as plain markdown so both humans and models can read, edit, and evolve it. The UI streams the loop live, and action execution is **explicitly approved** (or auto‑approved via a strict allowlist).

## Features ✅

- **Persistent memory** in markdown: experiences, knowledge, summaries, and actions
- **Live UI** powered by FastAPI + HTMX + SSE
- **Action approval flow** with optional auto‑execute (allowlist only)
- **Model‑agnostic brain** (SiliconFlow by default)
- **Human‑readable state** stored in `memory/state.md`

## Quick Start 🚀

1. Create a venv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment:

```bash
cp .env.example .env
# then edit .env
```

3. Run the server:

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

## Environment Variables 🧩

| Variable | Default | Description |
| --- | --- | --- |
| `SILICONFLOW_API_KEY` | — | API key (required) |
| `SILICONFLOW_MODEL` | `openai/gpt-oss-120b` | LLM model |
| `SILICONFLOW_BASE_URL` | `https://api.siliconflow.com/v1` | API base URL |
| `OPENULTRON_LOOP_INTERVAL` | `12` | Loop cadence (seconds) |
| `OPENULTRON_AUTO_EXECUTE` | `false` | Auto‑approve actions (allowlist only) |
| `OPENULTRON_SHELL_ALLOWLIST` | `ls,rg,cat,sed,python,python3,pip,git,uvicorn,pytest` | Allowed shell commands |

## How The Loop Works 🔁

`observe → think → act → evaluate → reflect → improve → repeat`

Each loop iteration writes a timestamped entry into `memory/experiences/YYYY-MM-DD.md`. The UI reads markdown files directly and streams updates via SSE.

## Action Execution 🛡️

Actions are proposed by the brain, stored in `memory/actions_queue.md`, and **must be approved** in the UI unless `OPENULTRON_AUTO_EXECUTE=true`.

Supported action types:
- `shell` (allowlisted commands only)
- `write_file`, `append_file`
- `write_memory`, `append_memory`
- `search_web`, `fetch_url`

Execution logs land in `memory/actions/YYYY-MM-DD.md`.

## Repo Map 🗂️

```text
openultron/
  actions.py        # action parsing + execution
  agent.py          # loop control
  brain.py          # model calls + response parsing
  memory.py         # memory IO helpers
  state.py          # loop state tracking
  utils.py          # shared helpers
memory/
  experiences/      # daily loop logs
  knowledge/        # distilled notes
  summaries/        # rollups
  actions/          # execution logs
  actions_queue.md  # pending actions
  state.md          # current state
templates/          # HTMX UI templates
app.py              # FastAPI server
```

## Development 🧪

Run tests:

```bash
pytest
```

## Roadmap 🧭

- Action sandboxing per skill
- Memory compression + retrieval ranking
- Multi‑agent collaboration
- Pluggable tool registry

## Contributing 🤝

See `CONTRIBUTING.md` for setup, conventions, and PR flow.

## Security 🔐

Please review `SECURITY.md` for reporting guidelines.

## License 📄

MIT — see `LICENSE`.
