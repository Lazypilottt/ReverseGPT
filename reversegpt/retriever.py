"""FAISS index over the tool docs, used to shortlist APIs for each prompt."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

    from reversegpt.registry import ApiRegistry

log = logging.getLogger(__name__)

FINGERPRINT_FILE = "tools.sha256"


def build_embeddings(model_name: str) -> Embeddings:
    """Local sentence-transformers embeddings (imported lazily, it pulls in torch)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})


class ToolRetriever:
    """Semantic search from a piece of text to the most relevant API names.

    Each tool is stored as a single document (name, description, arguments,
    outputs and example requests), so a hit always maps to exactly one API.
    """

    def __init__(self, store: FAISS):
        self._store = store

    @classmethod
    def build(cls, registry: ApiRegistry, embeddings: Embeddings) -> ToolRetriever:
        docs = [
            Document(page_content=spec.render(), metadata={"api_name": spec.name})
            for spec in registry
        ]
        return cls(FAISS.from_documents(docs, embeddings))

    def save(self, path: str | Path, fingerprint: str | None = None) -> None:
        path = Path(path)
        self._store.save_local(str(path))
        if fingerprint:
            (path / FINGERPRINT_FILE).write_text(fingerprint)

    @classmethod
    def load(cls, path: str | Path, embeddings: Embeddings) -> ToolRetriever:
        # The index is written by `reversegpt index` on this machine, so loading
        # the pickled docstore is safe here. Don't point this at untrusted files.
        store = FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
        return cls(store)

    @classmethod
    def load_or_build(
        cls, path: str | Path, registry: ApiRegistry, embeddings: Embeddings
    ) -> ToolRetriever:
        """Load the saved index, rebuilding it if missing or out of date."""
        path = Path(path)
        fingerprint = registry.fingerprint()
        stamp = path / FINGERPRINT_FILE
        if path.is_dir() and stamp.is_file() and stamp.read_text().strip() == fingerprint:
            return cls.load(path, embeddings)

        log.info("Tool docs changed or index missing, rebuilding index at %s", path)
        retriever = cls.build(registry, embeddings)
        retriever.save(path, fingerprint)
        return retriever

    def search(self, text: str, k: int = 5) -> list[str]:
        """API names ranked by similarity to ``text``."""
        hits = self._store.similarity_search(text, k=k)
        names: list[str] = []
        for doc in hits:
            name = doc.metadata.get("api_name")
            if name and name not in names:
                names.append(name)
        return names
