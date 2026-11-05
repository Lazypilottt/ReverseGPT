"""Command line interface: ``reversegpt tools | index | run``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import textwrap
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from reversegpt.backends import TOOLKITS
from reversegpt.config import Settings, load_settings
from reversegpt.registry import ApiRegistry

if TYPE_CHECKING:
    from reversegpt.pipeline import RunResult

NOISY_LOGGERS = ("httpx", "httpcore", "openai", "urllib3", "sentence_transformers", "faiss")


def _setup_logging(settings: Settings, verbose: bool) -> None:
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(settings.log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console], force=True)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def format_report(result: RunResult, registry: ApiRegistry, *, applied: bool) -> str:
    """Readable account of a run: each call, what it did, and the answer."""
    lines: list[str] = []
    for i, call in enumerate(result.calls):
        args = ", ".join(
            f"{name}={json.dumps(value, ensure_ascii=False)}"
            for name, value in call.arguments.items()
        )
        lines.append(f"[{i}] {call.api_name}({args})")
        for paragraph in call.message.splitlines():
            lines.extend(
                textwrap.wrap(paragraph, 96, initial_indent="    ", subsequent_indent="    ")
            )

    if not result.calls:
        lines.append("No API calls were made.")
    if result.status == "unrelated":
        lines.append("This request can't be handled with the tools in this toolkit.")
    if result.answer:
        lines += ["", result.answer]
    if result.status == "max_steps":
        lines += ["", "Stopped at max_plan_steps before the plan finished."]

    changes = sum(
        1 for c in result.calls if (spec := registry.get(c.api_name)) is not None and spec.writes
    )
    if changes and not applied:
        lines += [
            "",
            f"Dry run: {changes} change(s) previewed, nothing was modified. "
            "Re-run with --apply to make them.",
        ]
    elif changes:
        lines += ["", f"Applied {changes} change(s)."]
    return "\n".join(lines)


def cmd_tools(settings: Settings, _: argparse.Namespace) -> int:
    registry = ApiRegistry.from_directory(settings.tools_path)
    width = max(len(name) for name in registry.names())
    writes = any(spec.writes for spec in registry)
    for spec in registry:
        args = ", ".join(a.name + ("" if a.required else "?") for a in spec.arguments)
        mark = "* " if spec.writes else "  " if writes else ""
        print(f"{mark}{spec.name:<{width}}  ({args})")
    print(f"\n{len(registry)} tools in the {settings.toolkit} toolkit ({settings.tools_path})")
    print("? = optional" + ("   * = changes data, dry run unless --apply" if writes else ""))
    return 0


def cmd_index(settings: Settings, _: argparse.Namespace) -> int:
    from reversegpt.retriever import ToolRetriever, build_embeddings

    registry = ApiRegistry.from_directory(settings.tools_path)
    print(f"Embedding {len(registry)} tool docs with {settings.embedding_model} ...")
    retriever = ToolRetriever.build(registry, build_embeddings(settings.embedding_model))
    retriever.save(settings.index_path, registry.fingerprint())
    print(f"Index written to {settings.index_path}")
    return 0


def cmd_run(settings: Settings, args: argparse.Namespace) -> int:
    from reversegpt.pipeline import ReverseGPT

    query = " ".join(args.query).strip()
    if query == "-":
        query = sys.stdin.read().strip()
    if not query:
        print("error: empty query", file=sys.stderr)
        return 2

    _setup_logging(settings, args.verbose)
    logging.getLogger(__name__).info("Query: %s", query)

    agent = ReverseGPT.from_settings(settings, apply=args.apply)
    result = agent.run(query)
    tool_calls = result.tool_calls()

    output = Path(args.output) if args.output else settings.output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tool_calls, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(tool_calls, indent=2))
    else:
        print(format_report(result, agent.registry, applied=args.apply))
    print(f"\nSaved {len(tool_calls)} tool calls to {output}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reversegpt",
        description="Turn a plain-English request into a chain of API calls.",
    )
    parser.add_argument(
        "-c", "--config", default="config.ini", help="path to config file (default: config.ini)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--toolkit",
        choices=sorted(TOOLKITS),
        help="which tools to use: "
        + "; ".join(f"{name} = {about}" for name, about in TOOLKITS.items()),
    )
    common.add_argument(
        "--repo",
        metavar="OWNER/NAME",
        help="GitHub repository to work on (implies --toolkit github)",
    )

    sub.add_parser("tools", parents=[common], help="list the tools in the toolkit")
    sub.add_parser("index", parents=[common], help="(re)build the search index over the tool docs")

    run = sub.add_parser("run", parents=[common], help="plan and carry out a request")
    run.add_argument(
        "query", nargs="+", help='the request, in quotes, or "-" to read it from stdin'
    )
    run.add_argument(
        "--apply",
        action="store_true",
        help="really make changes (default: a dry run that only previews them)",
    )
    run.add_argument("--json", action="store_true", help="print the tool calls as JSON")
    run.add_argument("-o", "--output", help="where to write the tool calls JSON")
    run.add_argument(
        "-v", "--verbose", action="store_true", help="also print prompts and raw replies"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {"tools": cmd_tools, "index": cmd_index, "run": cmd_run}
    try:
        settings = load_settings(args.config)
        if args.repo:
            settings = dataclasses.replace(settings, toolkit="github", github_repo=args.repo)
        if args.toolkit:
            settings = dataclasses.replace(settings, toolkit=args.toolkit)
        return commands[args.command](settings, args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # show a clean message instead of a traceback
        if getattr(args, "verbose", False):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
