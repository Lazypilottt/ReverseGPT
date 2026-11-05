"""Chat model construction and helpers for talking to it."""

from __future__ import annotations

import ast
import json
import re
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from reversegpt.config import Settings


class ChatModel(Protocol):
    """Anything with a LangChain-style ``invoke``. Lets tests pass in a fake."""

    def invoke(self, input: str, **kwargs: Any) -> Any: ...


def build_chat_model(settings: Settings) -> ChatModel:
    """Create the chat model described by ``settings``.

    ``base_url`` makes it work with any OpenAI-compatible server
    (Ollama, vLLM, LM Studio, a corporate proxy, ...).
    """
    from langchain_openai import ChatOpenAI

    api_key = settings.api_key
    if not api_key:
        if not settings.base_url:
            raise RuntimeError(
                "No API key found. Set OPENAI_API_KEY or fill in api_key in config.ini."
            )
        api_key = "not-needed"  # local servers usually ignore it

    return ChatOpenAI(
        model=settings.model,
        temperature=settings.temperature,
        api_key=api_key,
        base_url=settings.base_url,
        timeout=60,
        max_retries=3,
    )


def complete(llm: ChatModel, prompt: str, stop: list[str] | None = None) -> str:
    """Send one prompt and return the reply text."""
    kwargs = {"stop": stop} if stop else {}
    reply = llm.invoke(prompt, **kwargs)
    text = getattr(reply, "content", reply)
    if isinstance(text, list):  # some providers return content blocks
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in text
        )
    return str(text).strip()


_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def parse_json(text: str) -> Any:
    """Pull the first JSON value out of a model reply.

    Models wrap JSON in code fences, put a sentence in front of it, or fall
    back to Python literals (single quotes, ``None``). All of those are handled
    here so the callers can stay simple.
    """
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for chunk in candidates:
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            pass
        for i, ch in enumerate(chunk):
            if ch not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(chunk, i)
                return value
            except json.JSONDecodeError:
                pass
            closing = "}" if ch == "{" else "]"
            end = chunk.rfind(closing)
            if end > i:
                try:
                    return ast.literal_eval(chunk[i : end + 1])
                except (ValueError, SyntaxError):
                    pass
    raise ValueError(f"no JSON found in model reply: {text[:200]!r}")
