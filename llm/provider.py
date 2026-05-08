"""
Provider factory — create the right LLMProvider from config or runtime settings.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DEFAULT_PROVIDER, GEMINI_API_KEY, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, GEMINI_MODEL


def create_provider(
    provider: str = "",
    api_key: str = "",
    model: str = "",
    base_url: str = "",
):
    """
    Create a provider by name.

    provider: "gemini" | "openai"  (default from DEFAULT_PROVIDER env)
    api_key / model / base_url: override the env-var defaults.
    """
    name = (provider or DEFAULT_PROVIDER).lower()

    if name == "gemini":
        from llm.gemini import GeminiProvider
        return GeminiProvider(
            api_key=api_key or GEMINI_API_KEY,
            model=model or GEMINI_MODEL,
        )
    elif name in ("openai", "openai_compat"):
        from llm.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(
            api_key=api_key or OPENAI_API_KEY,
            base_url=base_url or OPENAI_BASE_URL,
            model=model or OPENAI_MODEL,
        )
    else:
        raise ValueError(f"Unknown provider '{name}'. Valid: gemini, openai")
