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
