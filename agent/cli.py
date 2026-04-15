"""
OpenUltron CLI — Entrance Point
Usage:
  python3 -m agent.cli serve    # Start the web dashboard
  python3 -m agent.cli task     # Run a single task from terminal
"""
import asyncio
import sys
import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from agent.config import settings
from agent.core.engine import engine
from agent.memory.niche_db import init_db

app = typer.Typer(help="OpenUltron Agent CLI")
console = Console()

@app.command()
def serve(
    host: str = settings.host,
    port: int = settings.port,
    reload: bool = False
):
    """Start the OpenUltron Web Dashboard"""
    console.print(Panel(
        f"[bold red]OPENULTRON[/bold red] v0.1\n[dim]Experimental Agentic System[/dim]\n\n"
        f"Dashboard: [link=http://{host}:{port}]http://{host}:{port}[/link]\n"
        f"API Base: {settings.opencode_api_base}\n"
        f"Model: {settings.default_model}",
        title="[bold white]Initializing Server[/bold white]",
        border_style="red"
    ))
    
    uvicorn.run("web.main:app", host=host, port=port, reload=reload)

@app.command()
def run(
    description: str = typer.Argument(..., help="Task description"),
    incentive: str = typer.Option("", "--incentive", "-i", help="Incentive/Reward for success")
):
    """Run a single task directly in the terminal"""
    async def _run():
        await init_db()
        console.print(f"[bold blue]>[/bold blue] Starting task: [italic]{description}[/italic]")
        task_id = await engine.start_task(description, incentive)
        console.print(f"[bold blue]>[/bold blue] Task ID: [bold white]{task_id}[/bold white]")
        
        async for update in engine.run_loop():
            status = update.get("status")
            if status == "thinking":
                console.print(f"[dim]Thinking ({update.get('phase')})...[/dim]")
            elif status == "streaming":
                print(update.get("content"), end="", flush=True)
            elif status == "tool_executing":
                console.print(f"\n[bold yellow]TOOL CALL:[/bold yellow] {update.get('tool')}({update.get('args')})")
            elif status == "tool_result":
                res = update.get("result")
                color = "green" if res.get("success") else "red"
                console.print(f"[bold {color}]RESULT:[/bold {color}] {str(res)[:200]}...")
            elif status == "dreaming":
                console.print("\n[bold purple]DREAMING...[/bold purple] Consolidating memory.")
            elif status == "finished":
                state = update.get("state")
                outcome = state.get("outcome")
                color = "green" if outcome == "win" else "red"
                console.print(Panel(
                    f"Outcome: [bold {color}]{outcome.upper()}[/bold {color}]\n"
                    f"Result: {state.get('result')}\n"
                    f"Lessons: {', '.join(state.get('lessons_extracted', []))}",
                    title="Task Finished",
                    border_style=color
                ))
            elif status == "failed":
                console.print(f"\n[bold red]ERROR:[/bold red] {update.get('error')}")

    asyncio.run(_run())

if __name__ == "__main__":
    app()
