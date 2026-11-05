"""Settings loaded from an INI file, with environment variable fallbacks."""

from __future__ import annotations

import hashlib
import os
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path

from reversegpt.backends import TOOLKITS, toolkit_dir


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "reversegpt"


@dataclass
class Settings:
    # LLM
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    base_url: str | None = None

    # Tools
    toolkit: str = "devrev"
    docs_dir: Path | None = None  # default: the toolkit's bundled docs
    index_dir: Path | None = None  # default: a per-toolkit folder in the user cache

    # GitHub toolkit
    github_repo: str | None = None
    github_token: str | None = None
    github_api_url: str = "https://api.github.com"

    # Retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5

    # Agent loop
    max_plan_steps: int = 8
    max_chain_depth: int = 3

    # Output
    output_path: Path = Path("output/output.json")
    log_path: Path = Path("logs/run.log")

    @property
    def tools_path(self) -> Path:
        """Directory with the tool docs in use."""
        return self.docs_dir or toolkit_dir(self.toolkit)

    @property
    def index_path(self) -> Path:
        """Where the FAISS index for those docs is kept."""
        if self.index_dir:
            return self.index_dir
        if self.docs_dir:
            digest = hashlib.sha256(str(self.docs_dir.resolve()).encode()).hexdigest()[:10]
            return _cache_dir() / "index" / f"custom-{digest}"
        return _cache_dir() / "index" / self.toolkit


def load_settings(path: str | os.PathLike[str] | None = "config.ini") -> Settings:
    """Read settings from ``path``. A missing file just means "use the defaults".

    Secrets and endpoints fall back to the environment when the file leaves
    them empty: ``OPENAI_API_KEY``, ``OPENAI_BASE_URL`` and ``GITHUB_TOKEN``
    (or ``GH_TOKEN``, as set by the GitHub CLI).
    """
    parser = ConfigParser()
    if path is not None and Path(path).is_file():
        parser.read(path)

    def get(section: str, key: str) -> str | None:
        value = parser.get(section, key, fallback="").strip().strip('"').strip("'")
        return value or None

    def path_or_none(section: str, key: str) -> Path | None:
        value = get(section, key)
        return Path(value).expanduser() if value else None

    defaults = Settings()
    toolkit = (get("tools", "toolkit") or defaults.toolkit).lower()
    if toolkit not in TOOLKITS:
        raise ValueError(
            f"unknown toolkit {toolkit!r} in config; choose one of: {', '.join(TOOLKITS)}"
        )

    return Settings(
        api_key=get("llm", "api_key") or os.environ.get("OPENAI_API_KEY"),
        model=get("llm", "model") or defaults.model,
        temperature=float(get("llm", "temperature") or defaults.temperature),
        base_url=get("llm", "base_url") or os.environ.get("OPENAI_BASE_URL"),
        toolkit=toolkit,
        docs_dir=path_or_none("retriever", "docs_dir"),
        index_dir=path_or_none("retriever", "index_dir"),
        github_repo=get("github", "repo"),
        github_token=(
            get("github", "token") or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        ),
        github_api_url=get("github", "api_url") or defaults.github_api_url,
        embedding_model=get("retriever", "embedding_model") or defaults.embedding_model,
        top_k=int(get("retriever", "top_k") or defaults.top_k),
        max_plan_steps=int(get("agent", "max_plan_steps") or defaults.max_plan_steps),
        max_chain_depth=int(get("agent", "max_chain_depth") or defaults.max_chain_depth),
        output_path=Path(get("output", "output_path") or defaults.output_path),
        log_path=Path(get("output", "log_path") or defaults.log_path),
    )
