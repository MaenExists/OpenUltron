"""
The Agent Engine — Ultron's Core Execution Loop
Implements the 3-phase task structure: Understand → Act → Verify.
Manages context, tool execution, and the win/loss cycle.
"""
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import AsyncGenerator, Optional

from agent.config import settings
from agent.core.llm_client import llm
from agent.core.task_state import TaskState, TaskPhase, TaskOutcome
from agent.core.dreaming import dream
from agent.memory.structured_memory import read_all_memory
from agent.memory.niche_db import search_knowledge, record_task, update_task_outcome
from agent.identity.identity_manager import get_system_prompt
from agent.tools.sandbox_tools import TOOL_SCHEMAS, execute_tool

logger = logging.getLogger("openultron.engine")

class UltronEngine:
    """
    Core engine that drives the Ultron agent through its task phases.
    """
    
    def __init__(self):
        self.current_state: Optional[TaskState] = None
        self._stop_requested = False

    async def start_task(self, description: str, incentive: str = "") -> str:
        """Initialize a new task and return its ID."""
        state = TaskState(
            task_id=str(uuid.uuid4())[:8],
            description=description,
            incentive=incentive,
            phase=TaskPhase.PHASE_1_UNDERSTAND,
            started_at=time.time()
        )
        self.current_state = state
        self._stop_requested = False
        
        # Record in DB
        await record_task(state.task_id, description, incentive)
        
        # Initial context population
        await self._prepare_initial_context(state)
        
        return state.task_id

    async def _prepare_initial_context(self, state: TaskState):
        """Load long-term memories and search niche DB for relevant past experiences."""
        # 1. Load MD memory
        memories = await read_all_memory()
        mem_str = "\n\n".join([f"### {k.replace('_', ' ').title()}\n{v}" for k, v in memories.items()])
        
        # 2. Search Niche DB for similar tasks
        query_text = f"{state.description} {state.incentive}"
        try:
            # We use the LLM to get an embedding for the search query
            query_embedding = await llm.embed(query_text)
            similar = await search_knowledge(query_embedding, top_k=3)
            
            if similar:
                past_context = "\n".join([f"- Past lesson: {s['content']}" for s in similar])
                mem_str += f"\n\n### Relevant Past Experiences\n{past_context}"
        except Exception as e:
            logger.error(f"Niche DB search failed: {e}")

        # 3. Add to messages
        state.add_message("user", f"TASK START:\nDescription: {state.description}\nIncentive: {state.incentive}\n\nRELEVANT MEMORIES:\n{mem_str}")

    async def run_loop(self) -> AsyncGenerator[dict, None]:
        """
        The main execution loop. Streams state updates as JSON.
        This is what the UI connects to via SSE.
        """
        if not self.current_state:
            yield {"error": "No active task"}
            return

        state = self.current_state
        system_prompt = await get_system_prompt()
        
        try:
            while state.phase not in [TaskPhase.COMPLETE, TaskPhase.FAILED] and not self._stop_requested:
                # 1. Check for timeout
                if state.elapsed() > settings.task_timeout_seconds:
                    state.phase = TaskPhase.FAILED
                    state.outcome = TaskOutcome.TIMEOUT
                    state.error_context = "Task timed out."
                    break

                # 2. Call LLM for next action
                yield {"status": "thinking", "phase": state.phase.value}
                
                messages = state.get_messages_for_llm()
                
                # We use a tool-enabled chat call
                # Note: This is a simplified manual tool loop because we want to stream thoughts
                # but handle tool calls atomically.
                
                response_text = ""
                tool_calls = []

                # Stream the assistant's reasoning to the UI
                async for chunk in llm._client.chat.completions.create(
                    model=settings.default_model,
                    messages=[{"role": "system", "content": system_prompt}] + messages,
                    tools=TOOL_SCHEMAS,
                    stream=True
                ):
                    delta = chunk.choices[0].delta
                    if delta.content:
                        response_text += delta.content
                        yield {"status": "streaming", "content": delta.content}
                    
                    if delta.tool_calls:
                        # Accumulate tool calls
                        for tc in delta.tool_calls:
                            if len(tool_calls) <= tc.index:
                                tool_calls.append({"id": "", "name": "", "args": ""})
                            
                            if tc.id: tool_calls[tc.index]["id"] = tc.id
                            if tc.function and tc.function.name: tool_calls[tc.index]["name"] = tc.function.name
                            if tc.function and tc.function.arguments: tool_calls[tc.index]["args"] += tc.function.arguments

                # 3. Handle Assistant Response
                if response_text:
                    state.add_message("assistant", response_text)
                    # Capture output for dreaming
                    if state.phase == TaskPhase.PHASE_1_UNDERSTAND: state.phase1_output += response_text + "\n"
                    elif state.phase == TaskPhase.PHASE_2_ACT: state.phase2_output += response_text + "\n"
                    elif state.phase == TaskPhase.PHASE_3_VERIFY: state.phase3_output += response_text + "\n"

                # 4. Handle Tool Calls
                if tool_calls:
                    for tc in tool_calls:
                        name = tc["name"]
                        try:
                            args = json.loads(tc["args"])
                        except:
                            args = {}
                            
                        yield {"status": "tool_executing", "tool": name, "args": args}
                        
                        result = await execute_tool(name, args)
                        result_str = json.dumps(result)
                        
                        state.add_message("tool", result_str, tool_call_id=tc["id"], tool_name=name)
                        yield {"status": "tool_result", "tool": name, "result": result}
                        
                        # Increment API call counter
                        state.api_calls_used += 1

                # 5. Check for Phase Transitions (heuristic or explicit)
                # If the assistant says "MOVE TO PHASE X" or we detect a natural end
                if "WIN" in response_text or "LOSS" in response_text:
                    state.phase = TaskPhase.PHASE_3_VERIFY
                    if "WIN" in response_text: state.outcome = TaskOutcome.WIN
                    else: state.outcome = TaskOutcome.LOSS
                
                # Manual phase advancement if no tool call and no content (rare)
                if not response_text and not tool_calls:
                    if state.phase == TaskPhase.PHASE_1_UNDERSTAND: state.phase = TaskPhase.PHASE_2_ACT
                    elif state.phase == TaskPhase.PHASE_2_ACT: state.phase = TaskPhase.PHASE_3_VERIFY
                    else: state.phase = TaskPhase.COMPLETE

                # Logic to end the loop
                if state.outcome != TaskOutcome.PENDING and state.phase == TaskPhase.PHASE_3_VERIFY:
                    state.phase = TaskPhase.DREAMING
                    yield {"status": "dreaming"}
                    dream_results = await dream(state, llm.embed)
                    state.result = dream_results.get("summary", "Task completed.")
                    state.phase = TaskPhase.COMPLETE
                    state.completed_at = time.time()
                    break

            # Finalize
            await update_task_outcome(
                state.task_id, 
                state.outcome.value, 
                api_calls_used=state.api_calls_used,
                lessons=", ".join(state.lessons_extracted)
            )
            yield {"status": "finished", "state": state.to_dict()}

        except Exception as e:
            logger.exception(f"Engine loop crashed: {e}")
            state.phase = TaskPhase.FAILED
            state.error_context = str(e)
            yield {"status": "failed", "error": str(e)}

    def stop(self):
        self._stop_requested = True

# Global engine instance
engine = UltronEngine()
