import json

from conftest import DOCS, ScriptedLLM, StaticRetriever

from reversegpt import cli
from reversegpt.pipeline import ReverseGPT


def write_config(tmp_path):
    cfg = tmp_path / "config.ini"
    cfg.write_text(
        f"[retriever]\ndocs_dir = {DOCS}\n"
        f"[output]\noutput_path = {tmp_path / 'out.json'}\nlog_path = {tmp_path / 'run.log'}\n"
    )
    return cfg


def test_tools_command_lists_every_tool(tmp_path, capsys, registry):
    assert cli.main(["-c", str(write_config(tmp_path)), "tools"]) == 0
    out = capsys.readouterr().out
    for name in registry.names():
        assert name in out


def test_run_command_writes_output(tmp_path, capsys, monkeypatch, registry):
    llm = ScriptedLLM(
        plan=["Who am I?", "Final Answer: You are devu/1."],
        select={"Who am I?": "who_am_i"},
    )
    monkeypatch.setattr(
        ReverseGPT,
        "from_settings",
        classmethod(lambda cls, s: cls(llm, registry, StaticRetriever(registry), top_k=20)),
    )

    assert cli.main(["-c", str(write_config(tmp_path)), "run", "who", "am", "i"]) == 0

    expected = [{"tool_name": "who_am_i", "arguments": []}]
    assert json.loads((tmp_path / "out.json").read_text()) == expected
    assert json.loads(capsys.readouterr().out) == expected
    assert "Query: who am i" in (tmp_path / "run.log").read_text()
