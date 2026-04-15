"""
OpenUltron Web Server — FastAPI + HTMX + Tailwind CSS
Provides the agent dashboard and interaction layer.
"""
import asyncio
import json
import time
from typing import Optional
from fastapi import FastAPI, Request, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from agent.config import settings
from agent.core.engine import engine
from agent.memory.niche_db import get_task_history, get_win_loss_stats, init_db
from agent.identity.identity_manager import get_identity, get_identity_stats, initialize_identity
from agent.memory.structured_memory import read_all_memory

app = FastAPI(title="OpenUltron Dashboard")

# Mount static files (CSS/JS)
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Templates for HTMX
templates = Jinja2Templates(directory="web/templates")

# Initial setup
@app.on_event("startup")
async def startup_event():
    # Initialize DB and Identity if first run
    await init_db()
    # Check if identity exists, if not initialize default
    id_stats = await get_identity_stats()
    if id_stats["total_updates"] == 0 and id_stats["identity_length"] < 100:
        await initialize_identity(name=settings.agent_name)

# ─── Page Routes ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main Dashboard"""
    stats = await get_win_loss_stats()
    id_stats = await get_identity_stats()
    history = await get_task_history(limit=5)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "stats": stats,
        "id_stats": id_stats,
        "history": history,
        "agent_name": settings.agent_name,
        "current_task": engine.current_state
    })

@app.get("/memory", response_class=HTMLResponse)
async def memory_page(request: Request):
    """Memory Viewer"""
    memories = await read_all_memory()
    return templates.TemplateResponse("memory.html", {
        "request": request,
        "memories": memories
    })

@app.get("/identity", response_class=HTMLResponse)
async def identity_page(request: Request):
    """Identity Viewer"""
    identity_md = await get_identity()
    return templates.TemplateResponse("identity.html", {
        "request": request,
        "identity_md": identity_md
    })

# ─── API & HTMX Endpoints ────────────────────────────────────────────────────

class TaskRequest(BaseModel):
    description: str
    incentive: str = ""

@app.post("/tasks/start")
async def start_task(description: str = Form(...), incentive: str = Form("")):
    """Start a new agentic task"""
    if engine.current_state and engine.current_state.phase.value not in ["complete", "failed"]:
        raise HTTPException(status_code=400, detail="Task already in progress")
    
    task_id = await engine.start_task(description, incentive)
    return HTMLResponse(f'<div id="task-status" hx-get="/tasks/stream/{task_id}" hx-trigger="load" hx-swap="outerHTML">Starting task {task_id}...</div>')

@app.get("/tasks/stream/{task_id}")
async def stream_task(task_id: str):
    """SSE endpoint for real-time task progress"""
    if not engine.current_state or engine.current_state.task_id != task_id:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        async for update in engine.run_loop():
            # Wrap in SSE format
            yield f"data: {json.dumps(update)}\n\n"
            if update.get("status") == "finished":
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/stats")
async def get_stats(request: Request):
    """Fragment for auto-updating stats"""
    stats = await get_win_loss_stats()
    return templates.TemplateResponse("fragments/stats_grid.html", {"request": request, "stats": stats})

@app.get("/history")
async def get_history(request: Request):
    """Fragment for task history"""
    history = await get_task_history(limit=10)
    return templates.TemplateResponse("fragments/history_list.html", {"request": request, "history": history})

@app.post("/tasks/stop")
async def stop_task():
    """Request engine to stop"""
    engine.stop()
    return {"status": "stop requested"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
