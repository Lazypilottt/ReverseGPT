from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from reversegpt.registry import ApiRegistry

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "reversegpt" / "toolkits" / "devrev"
GITHUB_DOCS = ROOT / "reversegpt" / "toolkits" / "github"


@pytest.fixture(scope="session")
def registry() -> ApiRegistry:
    return ApiRegistry.from_directory(DOCS)


@pytest.fixture(scope="session")
def github_registry() -> ApiRegistry:
    return ApiRegistry.from_directory(GITHUB_DOCS)


class StaticRetriever:
    """Returns every tool, in registry order. Keeps pipeline tests independent of embeddings."""

    def __init__(self, registry: ApiRegistry):
        self.names = registry.names()

    def search(self, text: str, k: int = 5) -> list[str]:
        return self.names[:k]


class ScriptedLLM:
    """Fake chat model that answers each kind of prompt from a script.

    plan:       replies for the planner, consumed in order
    select:     plan step text -> api name
    arguments:  api name -> reply text (or a callable returning it)
    providers:  (api name, argument name) -> provider api name
    """

    def __init__(
        self,
        plan: list[str],
        select: dict[str, str] | None = None,
        arguments: dict[str, str | Callable[[str], str]] | None = None,
        providers: dict[tuple[str, str], str] | None = None,
    ):
        self.plan = list(plan)
        self.select = select or {}
        self.arguments = arguments or {}
        self.providers = providers or {}
        self.prompts: list[str] = []

    def invoke(self, prompt: str, **_: Any) -> str:
        self.prompts.append(prompt)

        if prompt.startswith("You plan how to answer"):
            return self.plan.pop(0) if self.plan else "Thought: done\nFinal Answer: done"

        if prompt.startswith("Pick the API that carries out"):
            step = re.search(r"Step to carry out now: (.*)", prompt).group(1).strip()
            return json.dumps({"api_name": self.select.get(step)})

        if prompt.startswith("Fill in the arguments"):
            api = re.search(r"^API: (\S+)", prompt, re.MULTILINE).group(1)
            reply = self.arguments.get(api, "{}")
            return reply(prompt) if callable(reply) else reply

        match = re.match(r'The API "(\S+)" needs the argument "(\S+)"', prompt)
        if match:
            return json.dumps({"api_name": self.providers.get(match.groups())})

        raise AssertionError(f"unexpected prompt:\n{prompt[:300]}")


@pytest.fixture
def make_agent(registry):
    from reversegpt.pipeline import ReverseGPT

    def factory(llm: ScriptedLLM, **kwargs: Any) -> ReverseGPT:
        kwargs.setdefault("top_k", len(registry))
        return ReverseGPT(llm, registry, StaticRetriever(registry), **kwargs)

    return factory
