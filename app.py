from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openultron.agent import Agent
from openultron.config import TEMPLATES_DIR, STATIC_DIR
from openultron.actions import (
    load_queue,
    approve_action,
    reject_action,
    run_next_approved,
    queue_actions,
)
from openultron.memory import list_memory_files, read_memory_file, latest_experience_entries
from openultron.state import read_state, set_status, set_goal, update_state
from openultron.runtime import load_runtime_settings, update_runtime_settings
from openultron.providers import (
    load_providers,
    get_active_provider,
    set_active_provider,
    upsert_provider,
    PROVIDER_PRESETS,
)

app = FastAPI()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

agent = Agent()


@app.on_event("startup")
async def startup() -> None:
    await agent.start_background()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    state = read_state()
    memory_files = list_memory_files()
    selected_path = memory_files[0] if memory_files else ""
    content = read_memory_file(selected_path)
    logs = latest_experience_entries()
    queue = load_queue()
    runtime_settings = load_runtime_settings()
    providers_data = load_providers()
    active_provider = get_active_provider()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "state": state,
            "memory_files": memory_files,
            "selected_path": selected_path,
            "memory_content": content,
            "logs": logs,
            "queue": queue,
            "settings": runtime_settings,
            "providers": providers_data.get("providers", []),
            "active_provider": active_provider,
            "provider_presets": PROVIDER_PRESETS,
        },
    )


@app.get("/partials/status", response_class=HTMLResponse)
async def partial_status(request: Request) -> HTMLResponse:
    state = read_state()
    return templates.TemplateResponse("partials/status.html", {"request": request, "state": state})


@app.get("/partials/logs", response_class=HTMLResponse)
async def partial_logs(request: Request) -> HTMLResponse:
    logs = latest_experience_entries()
    return templates.TemplateResponse("partials/logs_entries.html", {"request": request, "logs": logs})


@app.get("/stream/logs")
async def stream_logs(request: Request) -> StreamingResponse:
    async def event_generator():
        latest = latest_experience_entries(count=1)
        last_timestamp = latest[-1]["timestamp"] if latest else ""
        while True:
            if await request.is_disconnected():
                break
            entries = latest_experience_entries(count=1)
            if entries:
                entry = entries[-1]
                if entry["timestamp"] != last_timestamp:
                    last_timestamp = entry["timestamp"]
                    html = templates.get_template("partials/log_entry.html").render(entry=entry)
                    for line in html.splitlines():
                        yield f"data: {line}\n"
                    yield "\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/partials/memory", response_class=HTMLResponse)
async def partial_memory(request: Request, path: str = "") -> HTMLResponse:
    content = read_memory_file(path)
    return templates.TemplateResponse(
        "partials/memory.html",
        {"request": request, "memory_content": content, "selected_path": path},
    )


@app.get("/partials/queue", response_class=HTMLResponse)
async def partial_queue(request: Request) -> HTMLResponse:
    queue = load_queue()
    return templates.TemplateResponse("partials/queue.html", {"request": request, "queue": queue})


@app.get("/partials/settings", response_class=HTMLResponse)
async def partial_settings(request: Request) -> HTMLResponse:
    runtime_settings = load_runtime_settings()
    return templates.TemplateResponse(
        "partials/settings.html",
        {"request": request, "settings": runtime_settings},
    )


@app.post("/settings/update", response_class=HTMLResponse)
async def update_settings(request: Request) -> HTMLResponse:
    form = await request.form()
    auto_execute_values = form.getlist("auto_execute_actions")
    auto_execute = any(str(value).lower() == "true" for value in auto_execute_values)
    updates = {
        "loop_interval_seconds": form.get("loop_interval_seconds"),
        "max_actions_per_loop": form.get("max_actions_per_loop"),
        "max_loops": form.get("max_loops"),
        "max_stalls": form.get("max_stalls"),
        "shell_mode": form.get("shell_mode"),
        "shell_timeout_seconds": form.get("shell_timeout_seconds"),
        "shell_allowlist": form.get("shell_allowlist"),
        "auto_execute_actions": auto_execute,
    }
    runtime_settings = update_runtime_settings(updates)
    return templates.TemplateResponse(
        "partials/settings.html",
        {"request": request, "settings": runtime_settings},
    )


@app.get("/partials/providers", response_class=HTMLResponse)
async def partial_providers(request: Request) -> HTMLResponse:
    providers_data = load_providers()
    active_provider = get_active_provider()
    return templates.TemplateResponse(
        "partials/providers.html",
        {
            "request": request,
            "providers": providers_data.get("providers", []),
            "active_provider": active_provider,
            "provider_presets": PROVIDER_PRESETS,
        },
    )


@app.post("/providers/select", response_class=HTMLResponse)
async def select_provider(request: Request) -> HTMLResponse:
    form = await request.form()
    provider_id = str(form.get("provider_id", "")).strip()
    if provider_id:
        set_active_provider(provider_id)
    return await partial_providers(request)


@app.post("/providers/save", response_class=HTMLResponse)
async def save_provider(request: Request) -> HTMLResponse:
    form = await request.form()
    provider_id = str(form.get("provider_id", "")).strip()
    label = str(form.get("label", "")).strip()

    providers_data = load_providers()
    existing = None
    for provider in providers_data.get("providers", []):
        if provider_id and provider.get("id") == provider_id:
            existing = provider
            break
        if label and provider.get("label") == label:
            existing = provider
            break

    api_key = str(form.get("api_key", "")).strip() or (existing.get("api_key") if existing else "")

    headers: dict[str, str] = {}
    referer = str(form.get("openrouter_referer", "")).strip()
    title = str(form.get("openrouter_title", "")).strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    provider_payload = {
        "id": provider_id or (existing.get("id") if existing else ""),
        "label": label or (existing.get("label") if existing else ""),
        "base_url": str(form.get("base_url", "")).strip() or (existing.get("base_url") if existing else ""),
        "model": str(form.get("model", "")).strip() or (existing.get("model") if existing else ""),
        "api_key": api_key,
        "organization": str(form.get("organization", "")).strip() or (existing.get("organization") if existing else ""),
        "project": str(form.get("project", "")).strip() or (existing.get("project") if existing else ""),
        "headers": headers or (existing.get("headers") if existing else {}),
    }
    set_active = bool(form.get("set_active"))
    upsert_provider(provider_payload, set_active=set_active)
    return await partial_providers(request)


@app.post("/control/start", response_class=HTMLResponse)
async def control_start(request: Request) -> HTMLResponse:
    state = set_status("running")
    return templates.TemplateResponse("partials/status.html", {"request": request, "state": state})


@app.post("/control/pause", response_class=HTMLResponse)
async def control_pause(request: Request) -> HTMLResponse:
    state = set_status("paused")
    return templates.TemplateResponse("partials/status.html", {"request": request, "state": state})


@app.post("/control/step", response_class=HTMLResponse)
async def control_step(request: Request) -> HTMLResponse:
    await agent.loop_once()
    state = read_state()
    return templates.TemplateResponse("partials/status.html", {"request": request, "state": state})


@app.post("/control/goal", response_class=HTMLResponse)
async def control_goal(request: Request, goal: str = Form("")) -> HTMLResponse:
    state = set_goal(goal)
    return templates.TemplateResponse("partials/status.html", {"request": request, "state": state})


@app.post("/control/reset-error", response_class=HTMLResponse)
async def control_reset_error(request: Request) -> HTMLResponse:
    state = update_state(last_error="", last_action="Error cleared")
    return templates.TemplateResponse("partials/status.html", {"request": request, "state": state})


@app.post("/actions/approve", response_class=HTMLResponse)
async def action_approve(request: Request, action_id: str = Form("")) -> HTMLResponse:
    if action_id:
        approve_action(action_id)
    queue = load_queue()
    return templates.TemplateResponse("partials/queue.html", {"request": request, "queue": queue})


@app.post("/actions/reject", response_class=HTMLResponse)
async def action_reject(request: Request, action_id: str = Form("")) -> HTMLResponse:
    if action_id:
        reject_action(action_id)
    queue = load_queue()
    return templates.TemplateResponse("partials/queue.html", {"request": request, "queue": queue})


@app.post("/actions/run-next", response_class=HTMLResponse)
async def action_run_next(request: Request) -> HTMLResponse:
    await run_next_approved()
    queue = load_queue()
    return templates.TemplateResponse("partials/queue.html", {"request": request, "queue": queue})


@app.post("/actions/add", response_class=HTMLResponse)
async def action_add(
    request: Request,
    title: str = Form("Manual action"),
    action_type: str = Form("shell"),
    payload: str = Form("{}"),
) -> HTMLResponse:
    try:
        payload_data = json.loads(payload)
    except json.JSONDecodeError:
        payload_data = {}
    queue_actions(
        [
            {
                "title": title,
                "type": action_type,
                "payload": payload_data,
            }
        ],
        source="manual",
    )
    queue = load_queue()
    return templates.TemplateResponse("partials/queue.html", {"request": request, "queue": queue})
