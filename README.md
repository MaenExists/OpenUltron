# OpenUltron

OpenUltron is a persistent agent that lives in a filesystem, thinks with an LLM, learns from experience, and gradually improves itself through interaction with the world.

This build uses:
- Python + FastAPI for the server
- HTMX for live UI updates
- Tailwind CSS via CDN for the neon-obsidian visual system
- SiliconFlow API as the LLM brain (default model: `openai/gpt-oss-120b`)
- `python-multipart` for form handling
- Server-Sent Events (SSE) for the live experience feed

## Quick Start

1. Create a venv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Export your SiliconFlow key:

```bash
export SILICONFLOW_API_KEY="your_key_here"
```

Or create a `.env` file (recommended for local dev). Use `.env.example` as a template.

3. Run the server:

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

## Environment Variables

- `SILICONFLOW_API_KEY`: required to call SiliconFlow.
- `SILICONFLOW_MODEL`: optional, default `openai/gpt-oss-120b`.
- `SILICONFLOW_BASE_URL`: optional, default `https://api.siliconflow.com/v1`.
- `OPENULTRON_LOOP_INTERVAL`: loop cadence in seconds, default `12`.
- `OPENULTRON_AUTO_EXECUTE`: when `true`, auto-approves and runs up to 3 actions per loop.
- `OPENULTRON_SHELL_ALLOWLIST`: comma-separated shell commands allowed for action execution.

## How The Loop Works

The background loop runs:

`observe → think → act → evaluate → reflect → improve → repeat`

Each iteration writes a timestamped entry into `memory/experiences/YYYY-MM-DD.md`. State lives in `memory/state.md`. The UI reads these markdown files directly.

## Action Execution

The brain proposes actions (shell commands, file writes, web research) which are queued in `memory/actions_queue.md`. You approve actions in the UI, then execute them. Execution logs land in `memory/actions/YYYY-MM-DD.md`.

Supported action types:
- `shell` (allowlisted commands only)
- `write_file`, `append_file`
- `write_memory`, `append_memory`
- `search_web`, `fetch_url`

## Notes

Tailwind’s CDN build is intended for quick iteration. If you later want to productionize, switch to a build step for Tailwind.
