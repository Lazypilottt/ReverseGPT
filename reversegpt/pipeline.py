"""Ties the planner and Reverse Chain together into one agent loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from reversegpt.executor import PREV_REF, ExecutionError, Executor
from reversegpt.planner import Planner
from reversegpt.registry import ApiRegistry, ApiSpec, Argument
from reversegpt.reverse_chain import (
    ApiSelector,
    ArgumentExtractor,
    ProviderSelector,
    describe_value,
    is_missing,
)

if TYPE_CHECKING:
    from reversegpt.config import Settings
    from reversegpt.llm import ChatModel
    from reversegpt.retriever import ToolRetriever

log = logging.getLogger(__name__)


@dataclass
class ToolCall:
    api_name: str
    arguments: dict[str, Any]
    outputs: dict[str, Any]
    message: str

    @property
    def value(self) -> Any:
        """What $$PREV[i] points at: the call's first output, if any."""
        return next(iter(self.outputs.values()), None)


@dataclass
class RunResult:
    query: str
    calls: list[ToolCall] = field(default_factory=list)
    steps: list[tuple[str, str]] = field(default_factory=list)
    answer: str | None = None
    status: str = "done"  # done | unrelated | max_steps

    def tool_calls(self) -> list[dict[str, Any]]:
        """The calls in the DevRev answer format, with $$PREV[i] links kept."""
        return [
            {
                "tool_name": call.api_name,
                "arguments": [
                    {"argument_name": name, "argument_value": value}
                    for name, value in call.arguments.items()
                ],
            }
            for call in self.calls
        ]


class ReverseGPT:
    def __init__(
        self,
        llm: ChatModel,
        registry: ApiRegistry,
        retriever: ToolRetriever,
        executor: Executor | None = None,
        *,
        top_k: int = 5,
        max_plan_steps: int = 8,
        max_chain_depth: int = 3,
        planner_examples: str | None = None,
    ):
        self.registry = registry
        self.retriever = retriever
        self.executor = executor or Executor()
        self.planner = (
            Planner(llm, registry, examples=planner_examples)
            if planner_examples
            else Planner(llm, registry)
        )
        self.selector = ApiSelector(llm)
        self.extractor = ArgumentExtractor(llm)
        self.providers = ProviderSelector(llm, registry)
        self.top_k = top_k
        self.max_plan_steps = max_plan_steps
        self.max_chain_depth = max_chain_depth

    @classmethod
    def from_settings(cls, settings: Settings, *, apply: bool = False) -> ReverseGPT:
        """Build the agent for ``settings.toolkit``.

        ``apply`` lets tools that change data (create, assign, close, ...)
        actually do it. Without it they only report what they would do.
        """
        from reversegpt.backends import build_handlers, planner_examples
        from reversegpt.llm import build_chat_model
        from reversegpt.retriever import ToolRetriever, build_embeddings

        # Cheap checks first, so a missing key or repo fails before the model loads.
        llm = build_chat_model(settings)
        registry = ApiRegistry.from_directory(settings.tools_path)
        handlers = build_handlers(settings, apply=apply)

        embeddings = build_embeddings(settings.embedding_model)
        retriever = ToolRetriever.load_or_build(settings.index_path, registry, embeddings)
        return cls(
            llm,
            registry,
            retriever,
            Executor(handlers),
            top_k=settings.top_k,
            max_plan_steps=settings.max_plan_steps,
            max_chain_depth=settings.max_chain_depth,
            planner_examples=planner_examples(settings.tools_path),
        )

    # ------------------------------------------------------------------ loop

    def run(self, query: str) -> RunResult:
        result = RunResult(query=query)
        last_task: str | None = None
        last_api: str | None = None

        for _ in range(self.max_plan_steps):
            step = self.planner.next_step(query, result.steps)

            if step.unrelated:
                log.info("Planner: request can't be handled with these tools")
                result.status = "unrelated"
                return result
            if step.final:
                log.info("Planner: done")
                result.answer = step.answer
                return result

            exclude: set[str] = set()
            if step.retry and last_task:
                task = last_task
                if last_api:
                    exclude.add(last_api)
            elif step.text:
                task = step.text
            else:
                continue

            log.info("Step %d: %s", len(result.steps) + 1, task)
            observation, last_api = self._run_step(query, task, result.calls, exclude)
            log.info("  -> %s", observation)
            result.steps.append((task, observation))
            last_task = task

        log.warning("Stopped after %d plan steps", self.max_plan_steps)
        result.status = "max_steps"
        return result

    def _run_step(
        self, query: str, task: str, calls: list[ToolCall], exclude: set[str]
    ) -> tuple[str, str | None]:
        candidates = self._candidates(task, exclude=exclude)
        api_name = self.selector.select(query, task, candidates)
        spec = self.registry.get(api_name)
        if spec is None:
            return "No suitable API was found for this step.", None

        before = len(calls)
        try:
            self._resolve(query, task, spec, calls, depth=0, visiting=frozenset())
        except ExecutionError as exc:
            log.warning("%s failed: %s", spec.name, exc)
            done = self._describe_calls(calls, before)
            failure = f"{spec.name} could not run: {exc}."
            return (f"{done} Then {failure}" if done else failure), spec.name
        return self._describe_calls(calls, before), spec.name

    # ---------------------------------------------------------- reverse chain

    def _resolve(
        self,
        query: str,
        task: str,
        spec: ApiSpec,
        calls: list[ToolCall],
        depth: int,
        visiting: frozenset[str],
    ) -> int:
        """Run ``spec``, first running whatever produces its missing arguments.

        Returns the index of the call, i.e. the i in $$PREV[i].
        """
        visiting = visiting | {spec.name}
        arguments = self.extractor.extract(query, task, spec, self._variables(calls))
        arguments = {name: self._link_previous(value, calls) for name, value in arguments.items()}

        for arg in spec.required_arguments:
            if not is_missing(arguments.get(arg.name)):
                continue
            if depth >= self.max_chain_depth:
                log.warning("Chain depth limit reached while looking for %s", arg.name)
                continue

            provider = self._find_provider(query, spec, arg, visiting)
            if provider is None:
                log.info("  no API produces %s.%s", spec.name, arg.name)
                continue

            index = self._latest_call(calls, provider.name)
            if index is None:
                log.info("  %s needs %s -> running %s first", spec.name, arg.name, provider.name)
                sub_task = f"Get {arg.name} for {spec.name} ({arg.description})"
                index = self._resolve(query, sub_task, provider, calls, depth + 1, visiting)
            arguments[arg.name] = f"$$PREV[{index}]"

        arguments = {
            name: self._fit_type(spec.argument(name), value, calls)
            for name, value in arguments.items()
        }
        outputs, message = self.executor.execute(spec, arguments, [c.value for c in calls])
        calls.append(ToolCall(spec.name, arguments, outputs, message))
        return len(calls) - 1

    def _find_provider(
        self, query: str, consumer: ApiSpec, arg: Argument, visiting: frozenset[str]
    ) -> ApiSpec | None:
        candidates = self._candidates(f"{arg.name}: {arg.description}", exclude=set(visiting))
        candidates = [c for c in candidates if c.returns]
        return self.registry.get(self.providers.select(query, consumer, arg, candidates))

    # --------------------------------------------------------------- helpers

    def _candidates(self, text: str, exclude: set[str] | None = None) -> list[ApiSpec]:
        exclude = exclude or set()
        names = self.retriever.search(text, k=self.top_k + len(exclude))
        specs = [self.registry.get(n) for n in names if n not in exclude]
        return [s for s in specs if s is not None][: self.top_k]

    @staticmethod
    def _variables(calls: list[ToolCall]) -> str:
        lines = []
        for i, call in enumerate(calls):
            if not call.outputs:
                continue
            name = next(iter(call.outputs))
            lines.append(f"$$PREV[{i}]: {name} from {call.api_name} = {describe_value(call.value)}")
        return "\n".join(lines)

    @staticmethod
    def _describe_calls(calls: list[ToolCall], start: int) -> str:
        parts = []
        for i in range(start, len(calls)):
            call = calls[i]
            if call.outputs:
                name = next(iter(call.outputs))
                parts.append(f"{call.api_name} returned {name} = $$PREV[{i}]. {call.message}")
            else:
                parts.append(f"{call.api_name} ran successfully. {call.message}")
        return " ".join(parts)

    @staticmethod
    def _latest_call(calls: list[ToolCall], api_name: str) -> int | None:
        for i in range(len(calls) - 1, -1, -1):
            if calls[i].api_name == api_name and calls[i].outputs:
                return i
        return None

    @staticmethod
    def _link_previous(value: Any, calls: list[ToolCall]) -> Any:
        """Turn literal copies of earlier outputs back into $$PREV[i] references.

        The extractor sometimes pastes the value it was shown instead of the
        reference. The answer format wants the reference.
        """
        if is_missing(value) or (isinstance(value, str) and PREV_REF.match(value)):
            return value
        for i in range(len(calls) - 1, -1, -1):
            if calls[i].outputs and calls[i].value == value:
                return f"$$PREV[{i}]"
        if isinstance(value, list):
            return [ReverseGPT._link_previous(item, calls) for item in value]
        return value

    def _fit_type(self, arg: Argument | None, value: Any, calls: list[ToolCall]) -> Any:
        """Wrap a single value in a list when the argument expects an array.

        Referencing a scalar output (a user ID) for a list argument (owned_by)
        is written as ["$$PREV[0]"] in the answer format.
        """
        if arg is None or not arg.type.lower().startswith("array") or isinstance(value, list):
            return value
        if isinstance(value, str):
            match = PREV_REF.match(value)
            index = int(match.group(1)) if match else -1
            if 0 <= index < len(calls) and isinstance(calls[index].value, list):
                return value  # already points at a list
            return [value]
        return value
