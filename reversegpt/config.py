"""Settings loaded from an INI file, with environment variable fallbacks."""

from __future__ import annotations

import os
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    # LLM
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    base_url: str | None = None

    # Retrieval
    docs_dir: Path = Path("data/api_documentation")
    index_dir: Path = Path("vectorstore/db_faiss")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5

    # Agent loop
    max_plan_steps: int = 8
    max_chain_depth: int = 3

    # Output
    output_path: Path = Path("output/output.json")
    log_path: Path = Path("logs/run.log")


def load_settings(path: str | os.PathLike[str] | None = "config.ini") -> Settings:
    """Read settings from ``path``. A missing file just means "use the defaults".

    The API key is taken from the config file if set, otherwise from the
    ``OPENAI_API_KEY`` environment variable. ``OPENAI_BASE_URL`` works the same
    way for the endpoint.
    """
    parser = ConfigParser()
    if path is not None and Path(path).is_file():
        parser.read(path)

    def get(section: str, key: str) -> str | None:
        value = parser.get(section, key, fallback="").strip().strip('"').strip("'")
        return value or None

    defaults = Settings()
    s = Settings(
        api_key=get("llm", "api_key") or os.environ.get("OPENAI_API_KEY"),
        model=get("llm", "model") or defaults.model,
        temperature=float(get("llm", "temperature") or defaults.temperature),
        base_url=get("llm", "base_url") or os.environ.get("OPENAI_BASE_URL"),
        docs_dir=Path(get("retriever", "docs_dir") or defaults.docs_dir),
        index_dir=Path(get("retriever", "index_dir") or defaults.index_dir),
        embedding_model=get("retriever", "embedding_model") or defaults.embedding_model,
        top_k=int(get("retriever", "top_k") or defaults.top_k),
        max_plan_steps=int(get("agent", "max_plan_steps") or defaults.max_plan_steps),
        max_chain_depth=int(get("agent", "max_chain_depth") or defaults.max_chain_depth),
        output_path=Path(get("output", "output_path") or defaults.output_path),
        log_path=Path(get("output", "log_path") or defaults.log_path),
    )
    return s
