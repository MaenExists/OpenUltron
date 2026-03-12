# OpenUltron 🤖⚡

A persistent, filesystem‑native agent that **observes → thinks → acts → learns** and builds its own memory over time.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![HTMX](https://img.shields.io/badge/HTMX-UI-0F172A)
![License](https://img.shields.io/badge/License-MIT-22C55E)

## Why OpenUltron ✨

OpenUltron is built for **long‑running, self‑improving agent workflows**. It stores memory as plain markdown so both humans and models can read, edit, and evolve it. The UI streams the loop live, and action execution is **explicitly approved** (or auto‑approved via a strict allowlist).

## Features ✅

- **Provider‑agnostic LLM** via the OpenAI SDK (point `OPENAI_BASE_URL` to any OpenAI‑compatible provider)
- **Persistent memory** in markdown: experiences, knowledge, summaries, and actions
- **Live UI** powered by FastAPI + HTMX + SSE
- **Action approval flow** with optional auto‑execute
- **Loop guardrails** to prevent endless stalls
- **Provider wizard** to configure and switch LLM providers at runtime
- **Runtime settings panel** for loop cadence, stalls, and shell safety

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
| `OPENAI_API_KEY` | — | API key (required) |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API base URL for OpenAI‑compatible providers |
| `OPENAI_ORG` | — | Optional org ID |
| `OPENAI_PROJECT` | — | Optional project ID |
| `OPENULTRON_LOOP_INTERVAL` | `12` | Loop cadence (seconds) |
| `OPENULTRON_AUTO_EXECUTE` | `false` | Auto‑approve actions |
| `OPENULTRON_MAX_ACTIONS_PER_LOOP` | `5` | Execution cap per loop |
| `OPENULTRON_MAX_LOOPS` | `0` | Hard loop cap (0 = unlimited) |
| `OPENULTRON_MAX_STALLS` | `3` | Auto‑pause if stalling |
| `OPENULTRON_SHELL_MODE` | `allowlist` | `allowlist` or `full` |
| `OPENULTRON_SHELL_TIMEOUT` | `120` | Shell timeout seconds |
| `OPENULTRON_SHELL_ALLOWLIST` | `ls,rg,cat,sed,python,python3,pip,git,uvicorn,pytest` | Allowed shell commands |

## How The Loop Works 🔁

`observe → think → act → evaluate → reflect → improve → repeat`

Each loop iteration writes a timestamped entry into `memory/experiences/YYYY-MM-DD.md`. The UI reads markdown files directly and streams updates via SSE.

## Action Execution 🛡️

Actions are proposed by the brain, stored in `memory/actions_queue.md`, and **must be approved** in the UI unless `OPENULTRON_AUTO_EXECUTE=true`.

Supported action types:
- `shell` (allowlisted by default, full access if `OPENULTRON_SHELL_MODE=full`)
- `write_file`, `append_file`
- `write_memory`, `append_memory`
- `search_web`, `fetch_url`

Shell payload options:
- `cmd`: command string or list
- `cwd`: optional project‑relative working dir
- `timeout`: optional seconds
- `use_shell`: only when `OPENULTRON_SHELL_MODE=full`

## Memory System 🧠

- **Experiences**: loop logs per day
- **Knowledge**: lessons, unknowns, assumptions, failures
- **Skills**: reusable playbooks in `memory/skills/`

## Provider Wizard 🔌

Use the in‑app wizard to add or update providers and switch the active model without restarting the server.

Preset providers include:
- OpenAI
- OpenRouter
- Together AI
- Perplexity (v1 + v2)
- LM Studio (local)
- Ollama (local)
- vLLM (local)

The wizard auto‑fills base URLs and models for each preset, but everything remains editable. For OpenRouter, you can optionally set `HTTP-Referer` and `X-Title` headers in the wizard.

## Agent Settings 🎛️

Use the settings panel to update loop interval, auto‑execute, stall limits, and shell mode at runtime. These settings are stored in `memory/runtime_settings.json`.

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
  knowledge/        # lessons + unknowns + failures
  skills/           # playbooks and procedures
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
