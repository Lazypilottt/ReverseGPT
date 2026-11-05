"""The three LLM decisions behind Reverse Chain.

1. ApiSelector       - which API carries out a plan step (the "final" API).
2. ArgumentExtractor - which of its arguments can be filled from the request
                       or from earlier results.
3. ProviderSelector  - for each required argument that is still missing,
                       which API produces it. The pipeline then recurses on
                       that API, building the chain backwards.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from reversegpt.llm import ChatModel, complete, parse_json
from reversegpt.registry import ApiRegistry, ApiSpec, Argument

log = logging.getLogger(__name__)


def _render_candidates(specs: Sequence[ApiSpec]) -> str:
    return "\n\n".join(spec.render() for spec in specs)


def _pick_name(reply: str, allowed: Sequence[str]) -> str | None:
    """Read {"api_name": ...} from a reply and check it is one of ``allowed``."""
    try:
        data = parse_json(reply)
    except ValueError:
        data = reply.strip().strip('"').strip()
    name = data.get("api_name") if isinstance(data, dict) else data
    if isinstance(name, str) and name in allowed:
        return name
    if name not in (None, "None", "null", ""):
        log.warning("Model picked %r, which is not one of %s", name, list(allowed))
    return None


class ApiSelector:
    def __init__(self, llm: ChatModel):
        self.llm = llm

    def build_prompt(self, query: str, task: str, candidates: Sequence[ApiSpec]) -> str:
        return f"""\
Pick the API that carries out the step below.

User request: {query}
Step to carry out now: {task}

Candidate APIs:
{_render_candidates(candidates)}

Choose the API whose job matches the step itself, not one that only fetches an
input for it. Reply with JSON only: {{"api_name": "<one of the names above>"}},
or {{"api_name": null}} if none of them fit."""

    def select(self, query: str, task: str, candidates: Sequence[ApiSpec]) -> str | None:
        if not candidates:
            return None
        reply = complete(self.llm, self.build_prompt(query, task, candidates))
        return _pick_name(reply, [c.name for c in candidates])


class ArgumentExtractor:
    def __init__(self, llm: ChatModel):
        self.llm = llm

    def build_prompt(self, query: str, task: str, spec: ApiSpec, variables: str) -> str:
        return f"""\
Fill in the arguments for one call to the API documented below.

User request: {query}
Current step: {task}

API documentation:
{spec.render(examples=False)}

Results of earlier calls (refer to one by writing its reference, for example
"$$PREV[0]", as the argument value):
{variables or "none yet"}

Instructions:
- Reply with a JSON object that maps argument names to values.
- Take values from the request, the current step, or an earlier result.
- Use the types from the documentation; "array of strings" means a JSON list.
- If a required argument cannot be filled that way, set it to null. Another
  API will be found to produce it. Never invent IDs.
- Leave out optional arguments the request does not ask for.
- If the API takes no arguments, reply with {{}}."""

    def extract(self, query: str, task: str, spec: ApiSpec, variables: str = "") -> dict[str, Any]:
        if not spec.arguments:
            return {}
        reply = complete(self.llm, self.build_prompt(query, task, spec, variables))
        try:
            data = parse_json(reply)
        except ValueError:
            log.warning("Could not read arguments for %s from: %r", spec.name, reply)
            data = {}
        if not isinstance(data, dict):
            data = {}
        return clean_arguments(data, spec)


def clean_arguments(data: dict[str, Any], spec: ApiSpec) -> dict[str, Any]:
    """Drop unknown or empty arguments; keep required ones (as None if empty)."""
    cleaned: dict[str, Any] = {}
    for arg in spec.arguments:
        value = data.get(arg.name)
        if isinstance(value, str) and value.strip().lower() in (
            "",
            "none",
            "null",
            "requiredfalse",
        ):
            value = None
        if is_missing(value):
            if arg.required:
                cleaned[arg.name] = None
            continue
        cleaned[arg.name] = value
    unknown = set(data) - spec.argument_names
    if unknown:
        log.debug("Ignoring unknown arguments for %s: %s", spec.name, sorted(unknown))
    return cleaned


def is_missing(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


class ProviderSelector:
    def __init__(self, llm: ChatModel, registry: ApiRegistry):
        self.llm = llm
        self.registry = registry

    def build_prompt(
        self, query: str, consumer: ApiSpec, arg: Argument, candidates: Sequence[ApiSpec]
    ) -> str:
        return f"""\
The API "{consumer.name}" needs the argument "{arg.name}" ({arg.type}): {arg.description}
The value is not in the user's request, so it has to come from another API.

User request: {query}

Candidate APIs:
{_render_candidates(candidates)}

Which candidate returns a value that can be used as "{arg.name}"?
Reply with JSON only: {{"api_name": "<one of the names above>"}}, or
{{"api_name": null}} if none of them return it."""

    def select(
        self, query: str, consumer: ApiSpec, arg: Argument, candidates: Sequence[ApiSpec]
    ) -> str | None:
        if not candidates:
            return None

        # Shortcut: exactly one candidate returns an output with this exact
        # name (sprint_id -> get_sprint_id). No need to ask the model.
        exact = [c for c in candidates if any(o.name == arg.name for o in c.returns)]
        if len(exact) == 1:
            log.debug(
                "Provider for %s.%s by name match: %s", consumer.name, arg.name, exact[0].name
            )
            return exact[0].name

        reply = complete(self.llm, self.build_prompt(query, consumer, arg, candidates))
        return _pick_name(reply, [c.name for c in candidates])


def describe_value(value: Any, limit: int = 160) -> str:
    """Short, single-line rendering of a value for prompts and logs."""
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."
