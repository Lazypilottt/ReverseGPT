import json

import pytest

from reversegpt.config import load_settings
from reversegpt.executor import ExecutionError, Executor, resolve_references
from reversegpt.llm import parse_json
from reversegpt.planner import Planner
from reversegpt.registry import ApiRegistry, ApiSpec, ToolDocError
from reversegpt.reverse_chain import clean_arguments

# --------------------------------------------------------------- parse_json


@pytest.mark.parametrize(
    "reply, expected",
    [
        ('{"api_name": "who_am_i"}', {"api_name": "who_am_i"}),
        ('```json\n{"a": [1, 2]}\n```', {"a": [1, 2]}),
        ('Sure! Here it is: {"a": null} hope that helps', {"a": None}),
        ("{'work_id': 'X', 'flag': None}", {"work_id": "X", "flag": None}),
        ('["a", "b"]', ["a", "b"]),
        ("{}", {}),
    ],
)
def test_parse_json_handles_common_reply_shapes(reply, expected):
    assert parse_json(reply) == expected


def test_parse_json_raises_without_json():
    with pytest.raises(ValueError):
        parse_json("I don't know")


# ------------------------------------------------------------------ planner


@pytest.mark.parametrize(
    "reply, text, final, unrelated, retry",
    [
        ("Plan step 2: Get the sprint ID.", "Get the sprint ID.", False, False, False),
        ("Get the sprint ID.\nThought: then add items", "Get the sprint ID.", False, False, False),
        ("Thought: all done\nFinal Answer: It worked.", "", True, False, False),
        ("[]", "", False, True, False),
        ("Continue", "", False, False, True),
    ],
)
def test_planner_parse(reply, text, final, unrelated, retry):
    step = Planner.parse(reply)
    assert (step.text, step.final, step.unrelated, step.retry) == (text, final, unrelated, retry)


def test_planner_parse_extracts_final_answer():
    assert Planner.parse("Final Answer: Added 3 items.").answer == "Added 3 items."


def test_planner_prompt_numbers_steps(registry):
    from conftest import ScriptedLLM

    planner = Planner(ScriptedLLM(plan=[]), registry)
    prompt = planner.build_prompt("q", [("first", "ok"), ("second", "ok")])
    assert prompt.endswith(
        "Plan step 1: first\nAPI response: ok\nPlan step 2: second\nAPI response: ok\nPlan step 3:"
    )
    assert "- who_am_i:" in prompt


# --------------------------------------------------------- argument cleanup


def test_clean_arguments(registry):
    spec = registry.get("work_list")
    cleaned = clean_arguments(
        {"owned_by": ["u1"], "limit": "RequiredFalse", "stage.name": "", "bogus": 1}, spec
    )
    assert cleaned == {"owned_by": ["u1"]}

    spec = registry.get("add_work_items_to_sprint")
    assert clean_arguments({"work_ids": ["a"]}, spec) == {"work_ids": ["a"], "sprint_id": None}


# ----------------------------------------------------------------- executor


def test_resolve_references_splices_lists():
    previous = ["user-1", ["a", "b"]]
    assert resolve_references("$$PREV[0]", previous) == "user-1"
    assert resolve_references(["$$PREV[1]", "c"], previous) == ["a", "b", "c"]
    assert resolve_references(["$$PREV[0]"], previous) == ["user-1"]
    assert resolve_references("not a ref", previous) == "not a ref"
    with pytest.raises(ExecutionError):
        resolve_references("$$PREV[5]", previous)


def test_executor_requires_required_arguments(registry):
    with pytest.raises(ExecutionError, match="sprint_id"):
        Executor().execute(registry.get("add_work_items_to_sprint"), {"work_ids": ["a"]}, [])


def test_executor_maps_dotted_names(registry):
    outputs, _ = Executor().execute(
        registry.get("work_list"), {"ticket.severity": ["high"], "limit": 2}, []
    )
    assert len(outputs["work_ids"]) <= 2
    assert all(":ticket/" in w for w in outputs["work_ids"])


def test_executor_placeholder_for_tools_without_handler():
    spec = ApiSpec.from_dict(
        {"name": "send_email", "description": "d", "returns": [{"name": "message_id"}]}
    )
    outputs, message = Executor(handlers={}).execute(spec, {}, [])
    assert outputs == {"message_id": "<send_email.message_id>"}
    assert "send_email" in message


# ----------------------------------------------------------------- registry


def test_every_documented_tool_has_a_matching_mock(registry):
    """The mock server's output keys must match the `returns` names in the docs."""
    samples = {
        "work_id": "W-1",
        "query": "Acme",
        "objects": ["a", "b"],
        "text": "notes",
        "work_ids": ["a"],
        "sprint_id": "S-1",
    }
    for spec in registry:
        args = {a.name: samples[a.name] for a in spec.required_arguments}
        outputs, message = Executor().execute(spec, args, [])
        assert set(outputs) == {o.name for o in spec.returns}, spec.name
        assert message


def test_registry_rejects_bad_docs(tmp_path):
    (tmp_path / "a.json").write_text('{"name": "a", "description": "x"}')
    (tmp_path / "b.json").write_text('{"name": "a", "description": "y"}')
    with pytest.raises(ToolDocError, match="duplicate"):
        ApiRegistry.from_directory(tmp_path)

    (tmp_path / "b.json").write_text("{not json")
    with pytest.raises(ToolDocError, match="not valid JSON"):
        ApiRegistry.from_directory(tmp_path)

    with pytest.raises(ToolDocError, match="no tool docs"):
        ApiRegistry.from_directory(tmp_path / "empty")


def test_fingerprint_changes_with_docs(tmp_path):
    doc = {"name": "a", "description": "x"}
    (tmp_path / "a.json").write_text(json.dumps(doc))
    before = ApiRegistry.from_directory(tmp_path).fingerprint()
    doc["description"] = "y"
    (tmp_path / "a.json").write_text(json.dumps(doc))
    assert ApiRegistry.from_directory(tmp_path).fingerprint() != before


# ------------------------------------------------------------------- config


def test_settings_defaults_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    s = load_settings(tmp_path / "missing.ini")
    assert s.api_key == "sk-env"
    assert s.model == "gpt-4o-mini"

    cfg = tmp_path / "config.ini"
    cfg.write_text('[llm]\napi_key = sk-file\nmodel = "gpt-x"\n[agent]\nmax_plan_steps = 4\n')
    s = load_settings(cfg)
    assert (s.api_key, s.model, s.max_plan_steps) == ("sk-file", "gpt-x", 4)
