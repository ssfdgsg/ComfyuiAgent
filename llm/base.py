"""
Abstract LLM provider interface.

All providers convert from our internal message format to provider-specific
API calls and yield LLMEvent objects back.

Internal message format (provider-agnostic):
  messages = [
    {"role": "user",      "content": "text or list of parts"},
    {"role": "assistant", "content": "text"},
    {"role": "tool",      "tool_use_id": "...", "name": "...", "content": "..."},
  ]

Media files are passed as a list of MediaFile objects alongside the user message.
"""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator


@dataclasses.dataclass
class MediaFile:
    """A file to include in the next user message."""
    name: str
    mime_type: str   # e.g. "image/png", "audio/mp3", "video/mp4"
    data: bytes      # raw file bytes


@dataclasses.dataclass
class TextChunk:
    type: str = "token"
    content: str = ""


@dataclasses.dataclass
class ToolCallEvent:
    type: str = "tool_call"
    id: str = ""
    name: str = ""
    input: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DoneEvent:
    type: str = "done"
    content: str = ""   # full assembled text of the final turn


# Union type for all events
LLMEvent = TextChunk | ToolCallEvent | DoneEvent


class LLMProvider(ABC):
    """
    Base class for all LLM providers.

    `generate_stream` is the single entry point.
    It receives the full message history and yields events.
    The caller is responsible for appending tool results and calling again.
    """

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
        media_files: list[MediaFile] | None = None,
    ) -> AsyncGenerator[LLMEvent, None]:
        """
        Yield LLMEvent objects for one model turn.

        Stops after yielding all tool calls for the turn (caller executes them,
        appends results, and calls again).  When the model produces only text,
        yields TextChunk objects and a final DoneEvent.
        """
        ...

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def tool_result_message(tool_call_id: str, name: str, result: Any) -> dict:
        """Build a provider-agnostic tool-result message."""
        import json
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": result if isinstance(result, str) else json.dumps(result, ensure_ascii=False),
        }
