"""
Google Gemini provider.

Uses the `google-genai` SDK (pip install google-genai).
Supports:
  - Streaming text generation
  - Function / tool calling
  - Multimodal input: images, audio, video

Models tested:
  gemini-2.0-flash-exp  (default, fast, multimodal)
  gemini-1.5-pro        (better reasoning, higher cost)
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, AsyncGenerator

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llm.base import LLMProvider, LLMEvent, TextChunk, ToolCallEvent, DoneEvent, MediaFile
from config import GEMINI_API_KEY, GEMINI_MODEL, MAX_TOKENS, MAX_TOOL_ROUNDS


def _schema_to_gemini(schema: dict) -> dict:
    """
    Convert a JSON Schema dict to Gemini Schema dict.
    Gemini accepts the same JSON Schema format but needs some massaging
    (e.g., "default" fields must be removed, type arrays → single type).
    Returns a plain dict that Gemini's Schema() accepts via **kwargs.
    """
    out: dict = {}
    t = schema.get("type")
    if isinstance(t, list):
        t = t[0]  # take first type
    if t:
        out["type"] = t.upper()
    if "description" in schema:
        out["description"] = schema["description"]
    if "enum" in schema:
        out["enum"] = [str(e) for e in schema["enum"]]
    if t == "object" and "properties" in schema:
        out["properties"] = {
            k: _schema_to_gemini(v) for k, v in schema["properties"].items()
        }
        if "required" in schema:
            out["required"] = schema["required"]
    if t == "array" and "items" in schema:
        out["items"] = _schema_to_gemini(schema["items"])
    return out


def _tools_to_gemini(tools: list[dict]):
    """Convert our tool list to a Gemini Tool object."""
    from google.genai import types

    declarations = []
    for tool in tools:
        schema = tool.get("input_schema", {})
        params = _schema_to_gemini(schema) if schema.get("properties") else {"type": "OBJECT", "properties": {}}
        declarations.append(
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=params,
            )
        )
    return types.Tool(function_declarations=declarations)


def _build_contents(
    messages: list[dict],
    media_files: list[MediaFile] | None,
) -> list:
    """
    Convert our message list to Gemini contents list.

    Gemini roles: "user" | "model"
    Tool results are "user" role with function_response parts.
    """
    from google.genai import types

    contents = []
    # Group consecutive same-role messages
    for msg in messages:
        role = msg["role"]
        gemini_role = "model" if role == "assistant" else "user"

        if role == "tool":
            # Tool result → user turn with function_response
            content_str = msg.get("content", "")
            try:
                result_data = json.loads(content_str)
            except Exception:
                result_data = {"result": content_str}
            part = types.Part.from_function_response(
                name=msg.get("name", ""),
                response=result_data if isinstance(result_data, dict) else {"result": result_data},
            )
            contents.append(types.Content(role="user", parts=[part]))
            continue

        content = msg.get("content", "")
        parts = []

        if isinstance(content, str):
            if content:
                parts.append(types.Part.from_text(content))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(types.Part.from_text(item.get("text", "")))

        # Attach media files to the last user message
        if gemini_role == "user" and media_files and msg is messages[-1]:
            for mf in media_files:
                parts.append(types.Part.from_bytes(data=mf.data, mime_type=mf.mime_type))

        if parts:
            contents.append(types.Content(role=gemini_role, parts=parts))

    return contents


class GeminiProvider(LLMProvider):
    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        base_url: str = "",
    ):
        from google import genai
        from google.genai import types as _types

        key = api_key or GEMINI_API_KEY
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")

        client_kwargs: dict = {"api_key": key}
        if base_url:
            # Support proxy / custom Gemini-compatible endpoints
            client_kwargs["http_options"] = _types.HttpOptions(base_url=base_url)

        self._client = genai.Client(**client_kwargs)
        self._model = model or GEMINI_MODEL

    async def generate_stream(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
        media_files: list[MediaFile] | None = None,
    ) -> AsyncGenerator[LLMEvent, None]:
        from google.genai import types

        contents = _build_contents(messages, media_files)

        config_kwargs: dict[str, Any] = {
            "max_output_tokens": MAX_TOKENS,
        }
        if system:
            config_kwargs["system_instruction"] = system
        if tools:
            config_kwargs["tools"] = [_tools_to_gemini(tools)]

        gen_config = types.GenerateContentConfig(**config_kwargs)

        # Use streaming API
        response = self._client.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=gen_config,
        )

        full_text = ""
        tool_calls: list[ToolCallEvent] = []
        call_counter = 0

        for chunk in response:
            if not chunk.candidates:
                continue
            candidate = chunk.candidates[0]
            if not candidate.content or not candidate.content.parts:
                continue

            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    full_text += part.text
                    yield TextChunk(content=part.text)
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    call_id = f"call_{call_counter}"
                    call_counter += 1
                    args = dict(fc.args) if fc.args else {}
                    evt = ToolCallEvent(id=call_id, name=fc.name, input=args)
                    tool_calls.append(evt)
                    yield evt

        yield DoneEvent(content=full_text)
