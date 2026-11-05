import importlib.util
import os

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from reversegpt.retriever import FINGERPRINT_FILE, ToolRetriever


def test_build_save_load_roundtrip(registry, tmp_path):
    emb = DeterministicFakeEmbedding(size=32)
    index = tmp_path / "index"
    ToolRetriever.build(registry, emb).save(index, registry.fingerprint())

    loaded = ToolRetriever.load(index, emb)
    names = loaded.search("anything", k=len(registry))
    assert sorted(names) == sorted(registry.names())


def test_load_or_build_rebuilds_stale_index(registry, tmp_path):
    emb = DeterministicFakeEmbedding(size=32)
    index = tmp_path / "index"

    ToolRetriever.load_or_build(index, registry, emb)
    stamp = index / FINGERPRINT_FILE
    assert stamp.read_text() == registry.fingerprint()

    stamp.write_text("stale")
    ToolRetriever.load_or_build(index, registry, emb)
    assert stamp.read_text() == registry.fingerprint()


needs_hf = pytest.mark.skipif(
    importlib.util.find_spec("langchain_huggingface") is None or os.environ.get("CI") == "true",
    reason="needs langchain-huggingface and a model download",
)


@needs_hf
@pytest.mark.parametrize(
    "text, expected",
    [
        ("what is my user id", "who_am_i"),
        ("which sprint are we in right now", "get_sprint_id"),
        ("find issues that look like this one", "get_similar_work_items"),
        ("turn these meeting notes into tasks", "create_actionable_tasks_from_text"),
        ("look up the customer called Acme", "search_object_by_name"),
        ("give me a short overview of these tickets", "summarize_objects"),
    ],
)
def test_real_embeddings_rank_the_right_tool_first(registry, text, expected):
    from reversegpt.retriever import build_embeddings

    retriever = ToolRetriever.build(
        registry, build_embeddings("sentence-transformers/all-MiniLM-L6-v2")
    )
    assert retriever.search(text, k=3)[0] == expected
