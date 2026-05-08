"""
OpenAI-compatible provider.

Works with any endpoint that follows the OpenAI Chat Completions API:
  - OpenAI
  - ModelScope Inference API (https://api-inference.modelscope.cn/v1)
  - Ollama (local, http://localhost:11434/v1)
  - LM Studio, etc.

Multimodal: passes images as base64 data URLs in the message content.
Audio/Video: logged as text description (most compat endpoints don't support them).
"""
from __future__ import annotations

import base64
import json
import os
import sys
from typing import Any, AsyncGenerator

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llm.base import LLMProvider, LLMEvent, TextChunk, ToolCallEvent, DoneEvent, MediaFile
from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, MAX_TOKENS


def _build_openai_messages(
    messages: list[dict],
    system: str,
    media_files: list[MediaFile] | None,
) -> list[dict]:
    out = []
    if system:
        out.append({"role": "system", "content": system})

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg.get("content", "")

        if role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id", msg.get("tool_use_id", "")),
                "content": content,
            })
            continue

        if role == "assistant" and isinstance(content, list):
            # Handle tool_use blocks from previous turns
            tool_calls = []
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    tool_calls.append({
                        "id": item["id"],
                        "type": "function",
                        "function": {
                            "name": item["name"],
                            "arguments": json.dumps(item.get("input", {})),
                        },
                    })
                elif isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item["text"])
            openai_msg: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                openai_msg["content"] = "\n".join(text_parts)
            if tool_calls:
                openai_msg["tool_calls"] = tool_calls
            out.append(openai_msg)
            continue

        # Attach media files to the last user message
        is_last_user = (role == "user" and i == len(messages) - 1 and media_files)
        if is_last_user:
            parts: list[Any] = []
            if content:
                parts.append({"type": "text", "text": content})
            for mf in (media_files or []):
                if mf.mime_type.startswith("image/"):
                    b64 = base64.b64encode(mf.data).decode()
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mf.mime_type};base64,{b64}"},
                    })
                else:
                    parts.append({
                        "type": "text",
                        "text": f"[Attached {mf.mime_type} file: {mf.name}]",
                    })
            out.append({"role": "user", "content": parts})
        else:
            out.append({"role": role, "content": content})

    return out


def _tools_to_openai(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


class OpenAICompatProvider(LLMProvider):
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
    ):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=api_key or OPENAI_API_KEY or "no-key",
            base_url=base_url or OPENAI_BASE_URL,
        )
        self._model = model or OPENAI_MODEL

    async def generate_stream(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
        media_files: list[MediaFile] | None = None,
    ) -> AsyncGenerator[LLMEvent, None]:
        openai_messages = _build_openai_messages(messages, system, media_files)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": openai_messages,
            "max_tokens": MAX_TOKENS,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = _tools_to_openai(tools)
            kwargs["tool_choice"] = "auto"

        full_text = ""
        # tool_calls accumulator: id → {name, args_str}
        pending_calls: dict[int, dict] = {}

        async with self._client.chat.completions.stream(**kwargs) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                if delta.content:
                    full_text += delta.content
                    yield TextChunk(content=delta.content)

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in pending_calls:
                            pending_calls[idx] = {
                                "id": tc.id or f"call_{idx}",
                                "name": "",
                                "args": "",
                            }
                        if tc.function:
                            if tc.function.name:
                                pending_calls[idx]["name"] += tc.function.name
                            if tc.function.arguments:
                                pending_calls[idx]["args"] += tc.function.arguments

        # Emit accumulated tool calls
        for pc in pending_calls.values():
            try:
                args = json.loads(pc["args"]) if pc["args"] else {}
            except json.JSONDecodeError:
                args = {"_raw": pc["args"]}
            yield ToolCallEvent(id=pc["id"], name=pc["name"], input=args)

        yield DoneEvent(content=full_text)
