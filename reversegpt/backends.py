"""Toolkits: the docs for a set of tools plus the code that runs each tool."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from reversegpt import mock_server

if TYPE_CHECKING:
    from reversegpt.config import Settings
    from reversegpt.executor import Handler

log = logging.getLogger(__name__)

TOOLKITS_DIR = Path(__file__).resolve().parent / "toolkits"
TOOLKITS = {
    "devrev": "offline mock of the DevRev API (the original demo, needs no account)",
    "github": "your real GitHub issues",
}
PLANNER_EXAMPLES_FILE = "planner_examples.txt"


def toolkit_dir(name: str) -> Path:
    """Where the bundled docs for a toolkit live."""
    if name not in TOOLKITS:
        raise ValueError(f"unknown toolkit {name!r}; choose one of: {', '.join(TOOLKITS)}")
    return TOOLKITS_DIR / name


def planner_examples(docs_dir: Path) -> str | None:
    """Worked examples for the planner, if the docs directory has its own."""
    path = Path(docs_dir) / PLANNER_EXAMPLES_FILE
    return path.read_text(encoding="utf-8") if path.is_file() else None


def build_handlers(settings: Settings, *, apply: bool = False) -> Mapping[str, Handler]:
    """The functions that run each tool of ``settings.toolkit``."""
    if settings.toolkit == "github":
        return _github_handlers(settings, apply=apply)
    return mock_server.HANDLERS


def _github_handlers(settings: Settings, *, apply: bool) -> Mapping[str, Handler]:
    from reversegpt.github import GitHubClient, GitHubIssues

    if not settings.github_repo:
        raise ValueError(
            "No repository set. Pass --repo owner/name, or set repo under [github] in config.ini."
        )
    if apply and not settings.github_token:
        raise ValueError("--apply needs a GitHub token. Set GITHUB_TOKEN.")

    client = GitHubClient(
        settings.github_repo, token=settings.github_token, api_url=settings.github_api_url
    )
    repo = client.check_access()
    if not settings.github_token:
        log.warning(
            "No GitHub token: public repositories only, 60 requests an hour, and no writes. "
            "Set GITHUB_TOKEN for more."
        )
    mode = "changes WILL be made" if apply else "dry run, changes are only previewed"
    log.info("Connected to %s (%s).", repo.get("full_name", settings.github_repo), mode)
    return GitHubIssues(client, apply=apply).handlers()
