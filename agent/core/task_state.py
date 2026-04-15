"""
Task State Machine
Tracks the current state of Ultron's task execution.
Everything that can change during a task lives here.
"""
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskPhase(Enum):
    IDLE = "idle"
    PHASE_1_UNDERSTAND = "phase_1_understand"
    PHASE_2_ACT = "phase_2_act"
    PHASE_3_VERIFY = "phase_3_verify"
    DREAMING = "dreaming"
    COMPLETE = "complete"
    FAILED = "failed"


class TaskOutcome(Enum):
    PENDING = "pending"
    WIN = "win"
    LOSS = "loss"
    TIMEOUT = "timeout"


@dataclass
class ConversationMessage:
    role: str   # 'system', 'user', 'assistant', 'tool'
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None


@dataclass
class TaskState:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    incentive: str = ""

    # Execution state
    phase: TaskPhase = TaskPhase.IDLE
    outcome: TaskOutcome = TaskOutcome.PENDING
    retry_count: int = 0
    api_calls_used: int = 0

    # Conversation history (short-term memory = context window)
    messages: list[ConversationMessage] = field(default_factory=list)

    # Results and errors
    result: str = ""
    error_context: str = ""
    lessons_extracted: list[str] = field(default_factory=list)

    # Timing
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    # Phase outputs — captured for dreaming
    phase1_output: str = ""
    phase2_output: str = ""
    phase3_output: str = ""

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def add_message(self, role: str, content: str, **kwargs) -> None:
        self.messages.append(ConversationMessage(role=role, content=content, **kwargs))

    def get_messages_for_llm(self) -> list[dict]:
        """Convert to OpenAI message format."""
        result = []
        for msg in self.messages:
            m = {"role": msg.role, "content": msg.content}
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            result.append(m)
        return result

    def context_length(self) -> int:
        """Rough estimate of context window usage."""
        return sum(len(m.content) for m in self.messages)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "incentive": self.incentive,
            "phase": self.phase.value,
            "outcome": self.outcome.value,
            "retry_count": self.retry_count,
            "api_calls_used": self.api_calls_used,
            "result": self.result,
            "error_context": self.error_context,
            "lessons_extracted": self.lessons_extracted,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed": self.elapsed(),
        }
