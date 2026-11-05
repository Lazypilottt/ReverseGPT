"""Runs a tool call: resolves $$PREV references, validates, and dispatches."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from reversegpt import mock_server
from reversegpt.registry import ApiSpec
from reversegpt.reverse_chain import is_missing

log = logging.getLogger(__name__)

PREV_REF = re.compile(r"^\$\$PREV\[(\d+)\]$")

Handler = Callable[..., dict[str, Any]]


class ExecutionError(RuntimeError):
    pass


def resolve_references(value: Any, previous: Sequence[Any]) -> Any:
    """Replace "$$PREV[i]" strings (also inside lists) with the i-th call's output.

    A list output referenced inside a list is spliced in rather than nested,
    so ["$$PREV[0]"] with $$PREV[0] = ["a", "b"] becomes ["a", "b"].
    """
    if isinstance(value, str):
        match = PREV_REF.match(value.strip())
        if not match:
            return value
        index = int(match.group(1))
        if index >= len(previous):
            raise ExecutionError(f"{value} refers to a call that has not happened yet")
        return previous[index]
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            resolved = resolve_references(item, previous)
            if (
                isinstance(resolved, list)
                and isinstance(item, str)
                and PREV_REF.match(item.strip())
            ):
                out.extend(resolved)
            else:
                out.append(resolved)
        return out
    if isinstance(value, dict):
        return {k: resolve_references(v, previous) for k, v in value.items()}
    return value


def _python_name(argument_name: str) -> str:
    return argument_name.replace(".", "_")


class Executor:
    """Calls the handler registered for each API.

    By default the handlers are the mock DevRev endpoints in ``mock_server``.
    Pass your own mapping to call a real backend. APIs without a handler still
    "succeed" with placeholder outputs, so planning works for any documented
    tool even before it has an implementation.
    """

    def __init__(self, handlers: Mapping[str, Handler] | None = None):
        self.handlers = dict(mock_server.HANDLERS if handlers is None else handlers)

    def execute(
        self, spec: ApiSpec, arguments: Mapping[str, Any], previous: Sequence[Any]
    ) -> tuple[dict[str, Any], str]:
        resolved = {k: resolve_references(v, previous) for k, v in arguments.items()}

        missing = [a.name for a in spec.required_arguments if is_missing(resolved.get(a.name))]
        if missing:
            raise ExecutionError(f"missing required argument(s): {', '.join(missing)}")

        handler = self.handlers.get(spec.name)
        if handler is None:
            outputs = {o.name: f"<{spec.name}.{o.name}>" for o in spec.returns}
            return outputs, f"Called {spec.name}."

        try:
            result = dict(handler(**{_python_name(k): v for k, v in resolved.items()}))
        except TypeError as exc:  # wrong arguments for the handler
            raise ExecutionError(str(exc)) from exc

        message = str(result.pop("message", f"Called {spec.name}."))
        return result, message
