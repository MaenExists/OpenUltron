"""
LLM Client — wraps OpenCode Zen API (OpenAI-compatible)
Handles streaming, retries, token counting, multi-model routing.
"""
import asyncio
import json
import time
from typing import AsyncGenerator, Optional
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from agent.config import settings


class LLMClient:
    """
    Central LLM interface. Uses OpenCode Zen (opencode.ai/zen/v1) 
    which is OpenAI-compatible and gives us access to Claude, Gemini, GPT etc.
    """

    def __init__(self):
        self._client = AsyncOpenAI(
            api_key=settings.opencode_api_key,
            base_url=settings.opencode_api_base,
            timeout=settings.api_timeout_seconds,
        )
        self._call_count = 0

    def reset_call_count(self):
        self._call_count = 0

    async def chat(
        self,
        messages: list[ChatCompletionMessageParam],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = 0.7,
        system: Optional[str] = None,
    ) -> str:
        """
        Simple one-shot chat completion. Returns full response string.
        Enforces API call limit per task.
        """
        if self._call_count >= settings.max_api_calls_per_task:
            raise RuntimeError(
                f"Hit API call limit ({settings.max_api_calls_per_task}) for this task. "
                "Resetting or completing the task first."
            )

        model = model or settings.default_model
        max_tokens = max_tokens or settings.max_tokens_per_request

        # Prepend system message if provided
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        self._call_count += 1

        try:
            resp = await self._client.chat.completions.create(
                model=model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"LLM call failed (model={model}): {e}") from e

    async def stream(
        self,
        messages: list[ChatCompletionMessageParam],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = 0.7,
        system: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming chat completion — yields text chunks as they arrive.
        Used by the SSE endpoint to stream agent reasoning to the UI.
        """
        if self._call_count >= settings.max_api_calls_per_task:
            raise RuntimeError("Hit API call limit for this task.")

        model = model or settings.default_model
        max_tokens = max_tokens or settings.max_tokens_per_request

        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        self._call_count += 1

        try:
            stream = await self._client.chat.completions.create(
                model=model,
                messages=full_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as e:
            raise RuntimeError(f"LLM stream failed (model={model}): {e}") from e

    async def chat_with_retry(
        self,
        messages: list[ChatCompletionMessageParam],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        retries: int = 3,
        system: Optional[str] = None,
    ) -> str:
        """
        Chat with exponential backoff retry. Used for critical operations.
        """
        last_error = None
        for attempt in range(retries):
            try:
                return await self.chat(messages, model=model, max_tokens=max_tokens, system=system)
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s
        raise RuntimeError(f"LLM call failed after {retries} retries: {last_error}")

    async def embed(self, text: str, model: str = "opencode/gemini-3-flash") -> list[float]:
        """
        Generate text embeddings for semantic memory search.
        Falls back to a simple hash-based pseudo-embedding if the API doesn't support it.
        """
        try:
            resp = await self._client.embeddings.create(
                model=model,
                input=text,
            )
            return resp.data[0].embedding
        except Exception:
            # Fallback: deterministic pseudo-embedding using char frequencies
            # Not great for semantic search but won't crash the system
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            # 64 floats from the hash bytes
            return [(b / 255.0) * 2 - 1 for b in h * 4][:64]


# Global client instance — shared across the agent
llm = LLMClient()
