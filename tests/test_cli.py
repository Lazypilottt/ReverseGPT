import json

from conftest import DOCS, ScriptedLLM, StaticRetriever

from reversegpt import cli
from reversegpt.pipeline import ReverseGPT, RunResult, ToolCall


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


def test_repo_flag_switches_to_the_github_toolkit(tmp_path, capsys, github_registry):
    assert cli.main(["-c", str(tmp_path / "none.ini"), "tools", "--repo", "acme/app"]) == 0
    out = capsys.readouterr().out
    assert "11 tools in the github toolkit" in out
    assert "* assign_issues" in out  # write tools are marked
    for name in github_registry.names():
        assert name in out


def test_run_command_writes_output(tmp_path, capsys, monkeypatch, registry):
    llm = ScriptedLLM(
        plan=["Who am I?", "Final Answer: You are devu/1."],
        select={"Who am I?": "who_am_i"},
    )
    seen = {}

    def fake_from_settings(cls, settings, **kwargs):
        seen.update(kwargs)
        return cls(llm, registry, StaticRetriever(registry), top_k=20)

    monkeypatch.setattr(ReverseGPT, "from_settings", classmethod(fake_from_settings))

    cfg = str(write_config(tmp_path))
    assert cli.main(["-c", cfg, "run", "--json", "who", "am", "i"]) == 0

    expected = [{"tool_name": "who_am_i", "arguments": []}]
    assert json.loads((tmp_path / "out.json").read_text()) == expected
    assert json.loads(capsys.readouterr().out) == expected
    assert "Query: who am i" in (tmp_path / "run.log").read_text()
    assert seen == {"apply": False}


def test_report_shows_each_call_and_the_dry_run_notice(github_registry):
    result = RunResult(
        query="assign my bugs",
        calls=[
            ToolCall("search_issues", {"labels": ["bug"]}, {"issue_numbers": [1]}, "Found 1."),
            ToolCall("assign_issues", {"issue_numbers": "$$PREV[0]"}, {}, "[dry run] Would..."),
        ],
        answer="Done.",
    )
    report = cli.format_report(result, github_registry, applied=False)
    assert report.splitlines()[:2] == ['[0] search_issues(labels=["bug"])', "    Found 1."]
    assert '[1] assign_issues(issue_numbers="$$PREV[0]")' in report
    assert "Dry run: 1 change(s) previewed" in report

    applied = cli.format_report(result, github_registry, applied=True)
    assert "Applied 1 change(s)." in applied and "Dry run" not in applied
