"""Command line interface: ``reversegpt index | tools | run``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

from reversegpt.config import Settings, load_settings
from reversegpt.registry import ApiRegistry

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


def cmd_tools(settings: Settings, _: argparse.Namespace) -> int:
    registry = ApiRegistry.from_directory(settings.docs_dir)
    width = max(len(name) for name in registry.names())
    for spec in registry:
        args = ", ".join(a.name + ("" if a.required else "?") for a in spec.arguments)
        print(f"{spec.name:<{width}}  ({args})")
    print(f"\n{len(registry)} tools in {settings.docs_dir}  (? = optional)")
    return 0


def cmd_index(settings: Settings, _: argparse.Namespace) -> int:
    from reversegpt.retriever import ToolRetriever, build_embeddings

    registry = ApiRegistry.from_directory(settings.docs_dir)
    print(f"Embedding {len(registry)} tool docs with {settings.embedding_model} ...")
    retriever = ToolRetriever.build(registry, build_embeddings(settings.embedding_model))
    retriever.save(settings.index_dir, registry.fingerprint())
    print(f"Index written to {settings.index_dir}")
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

    agent = ReverseGPT.from_settings(settings)
    result = agent.run(query)
    tool_calls = result.tool_calls()

    output = Path(args.output) if args.output else settings.output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tool_calls, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(tool_calls, indent=2))
    if result.answer:
        print(f"\n{result.answer}", file=sys.stderr)
    if result.status == "max_steps":
        print("\nwarning: stopped at max_plan_steps before the plan finished", file=sys.stderr)
    print(f"\nSaved {len(tool_calls)} tool calls to {output}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reversegpt",
        description="Turn a natural-language request into a chain of API calls.",
    )
    parser.add_argument(
        "-c", "--config", default="config.ini", help="path to config file (default: config.ini)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("tools", help="list the tools found in the docs directory")
    sub.add_parser("index", help="(re)build the FAISS index over the tool docs")

    run = sub.add_parser("run", help="plan and execute a request")
    run.add_argument(
        "query", nargs="+", help='the request, in quotes, or "-" to read it from stdin'
    )
    run.add_argument("-o", "--output", help="where to write the tool calls JSON")
    run.add_argument(
        "-v", "--verbose", action="store_true", help="also print prompts and raw replies"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    commands = {"tools": cmd_tools, "index": cmd_index, "run": cmd_run}
    try:
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
