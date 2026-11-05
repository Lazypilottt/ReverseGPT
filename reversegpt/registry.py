"""Tool documentation: loading, validation and rendering for prompts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


class ToolDocError(ValueError):
    """Raised when a tool documentation file is missing fields or malformed."""


@dataclass(frozen=True)
class Argument:
    name: str
    type: str
    description: str
    required: bool = False


@dataclass(frozen=True)
class Output:
    name: str
    type: str
    description: str


@dataclass(frozen=True)
class ApiSpec:
    name: str
    description: str
    arguments: tuple[Argument, ...] = ()
    returns: tuple[Output, ...] = ()
    examples: tuple[str, ...] = ()
    writes: bool = False  # changes data; such tools run as a dry run unless applied

    @property
    def required_arguments(self) -> tuple[Argument, ...]:
        return tuple(a for a in self.arguments if a.required)

    @property
    def argument_names(self) -> set[str]:
        return {a.name for a in self.arguments}

    def argument(self, name: str) -> Argument | None:
        return next((a for a in self.arguments if a.name == name), None)

    def render(self, *, examples: bool = True) -> str:
        """Plain-text description used both for embeddings and inside prompts."""
        lines = [f"API: {self.name}", f"Description: {self.description}"]
        if self.arguments:
            lines.append("Arguments:")
            for a in self.arguments:
                flag = "required" if a.required else "optional"
                lines.append(f"  - {a.name} ({a.type}, {flag}): {a.description}")
        else:
            lines.append("Arguments: none")
        if self.returns:
            lines.append("Returns:")
            lines.extend(f"  - {o.name} ({o.type}): {o.description}" for o in self.returns)
        else:
            lines.append("Returns: nothing")
        if examples and self.examples:
            lines.append("Example requests: " + "; ".join(self.examples))
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict, *, source: str = "<dict>") -> ApiSpec:
        try:
            return cls(
                name=data["name"],
                description=data["description"],
                arguments=tuple(
                    Argument(
                        name=a["name"],
                        type=a.get("type", "string"),
                        description=a.get("description", ""),
                        required=bool(a.get("required", False)),
                    )
                    for a in data.get("arguments", [])
                ),
                returns=tuple(
                    Output(
                        name=o["name"],
                        type=o.get("type", "string"),
                        description=o.get("description", ""),
                    )
                    for o in data.get("returns", [])
                ),
                examples=tuple(data.get("examples", [])),
                writes=bool(data.get("writes", False)),
            )
        except (KeyError, TypeError) as exc:
            raise ToolDocError(f"{source}: invalid tool doc ({exc!r})") from exc


class ApiRegistry:
    """All known tools, keyed by name."""

    def __init__(self, specs: Iterable[ApiSpec]):
        self._specs: dict[str, ApiSpec] = {}
        for spec in specs:
            if spec.name in self._specs:
                raise ToolDocError(f"duplicate tool name: {spec.name}")
            self._specs[spec.name] = spec

    @classmethod
    def from_directory(cls, path: str | Path) -> ApiRegistry:
        path = Path(path)
        files = sorted(path.glob("*.json"))
        if not files:
            raise ToolDocError(f"no tool docs (*.json) found in {path}")
        specs = []
        for file in files:
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ToolDocError(f"{file}: not valid JSON ({exc})") from exc
            specs.append(ApiSpec.from_dict(data, source=str(file)))
        return cls(specs)

    def get(self, name: str | None) -> ApiSpec | None:
        return self._specs.get(name) if name else None

    def names(self) -> list[str]:
        return list(self._specs)

    def fingerprint(self) -> str:
        """Hash of every spec, used to notice when the search index is stale."""
        blob = "\n\n".join(self._specs[n].render() for n in sorted(self._specs))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __iter__(self) -> Iterator[ApiSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)
