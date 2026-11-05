"""GitHub toolkit: query building, dry runs, ranking and the full loop, against a fake GitHub."""

from __future__ import annotations

import inspect
import io
import json
import re
import urllib.error
from datetime import datetime, timezone
from typing import Any

import pytest
from conftest import ScriptedLLM, StaticRetriever

from reversegpt.github import (
    SIMILARITY_THRESHOLD,
    GitHubClient,
    GitHubError,
    GitHubIssues,
    _similarity,
)
from reversegpt.pipeline import ReverseGPT

NOW = datetime(2026, 11, 5, tzinfo=timezone.utc)
REPO = "acme/app"


def issue(number, title, labels=(), state="open", assignees=(), comments=0, reactions=0,
          created="2026-10-01T00:00:00Z"):  # fmt: skip
    return {
        "number": number,
        "title": title,
        "state": state,
        "labels": [{"name": name} for name in labels],
        "assignees": [{"login": a} for a in assignees],
        "comments": comments,
        "reactions": {"total_count": reactions},
        "created_at": created,
    }


class FakeGitHub:
    """In-memory stand-in for the GitHub REST API, enough for these tools."""

    def __init__(self, issues, milestones=(), labels=("bug", "p0", "p1"), token="t"):
        self.repo = REPO
        self.token = token
        self.issues = {i["number"]: i for i in issues}
        self.milestones = {m["number"]: m for m in milestones}
        self.labels = list(labels)
        self.calls: list[tuple[str, str, Any]] = []

    @property
    def writes(self):
        return [c for c in self.calls if c[0] != "GET"]

    def request(self, method, path, *, params=None, body=None):
        self.calls.append((method, path, params if method == "GET" else body))
        base = f"/repos/{REPO}"
        if path == "/user":
            return {"login": "octocat"}
        if path == "/search/issues":
            return self._search(params["q"], int(params.get("per_page", 30)))
        if path == f"{base}/milestones":
            return list(self.milestones.values())
        if m := re.fullmatch(rf"{base}/milestones/(\d+)", path):
            return self._get(self.milestones, int(m.group(1)))
        if path == f"{base}/labels":
            return [{"name": n} for n in self.labels]
        if path == f"{base}/issues" and method == "POST":
            number = max(self.issues, default=0) + 1
            self.issues[number] = issue(number, body["title"], body.get("labels", ()))
            return self.issues[number]
        if m := re.fullmatch(rf"{base}/issues/(\d+)(/\w+)?", path):
            return self._get(self.issues, int(m.group(1)))
        raise AssertionError(f"unexpected request {method} {path}")

    @staticmethod
    def _get(table, key):
        if key not in table:
            raise GitHubError(f"GitHub found nothing at #{key}")
        return table[key]

    def _search(self, q, per_page):
        items = list(self.issues.values())
        if "is:open" in q:
            items = [i for i in items if i["state"] == "open"]
        for label in re.findall(r'label:"([^"]+)"', q):
            items = [i for i in items if label in [x["name"] for x in i["labels"]]]
        if "no:assignee" in q:
            items = [i for i in items if not i["assignees"]]
        for login in re.findall(r"assignee:(\S+)", q):
            items = [i for i in items if login in [a["login"] for a in i["assignees"]]]
        if group := re.search(r"\(([^)]*)\) in:title", q):
            words = [w.strip() for w in group.group(1).split(" OR ")]
            items = [i for i in items if any(w in i["title"].lower() for w in words)]
        return {"total_count": len(items), "items": items[:per_page]}


@pytest.fixture
def gh():
    return FakeGitHub(
        [
            issue(1, "Login times out on slow networks", ["bug"]),
            issue(2, "Login timeout on slow connections", ["bug", "p1"], assignees=["dev"]),
            issue(3, "Add dark mode", ["enhancement"], reactions=40),
            issue(4, "Data loss when sync fails", ["bug", "p0"], comments=12),
            issue(
                5,
                "Old question about setup",
                ["question"],
                state="closed",
                created="2025-01-01T00:00:00Z",
            ),  # fmt: skip
        ],
        milestones=[
            {"number": 7, "title": "v2.1", "state": "open", "due_on": "2026-12-01T00:00:00Z"},
            {"number": 6, "title": "v2.0", "state": "open", "due_on": "2026-11-20T00:00:00Z"},
            {"number": 8, "title": "Backlog", "state": "open", "due_on": None},
        ],
    )


def tools(client, apply=False):
    return GitHubIssues(client, apply=apply, now=lambda: NOW)


# ------------------------------------------------------------------ toolkit


def test_every_documented_tool_has_a_handler_that_accepts_its_arguments(github_registry, gh):
    handlers = tools(gh).handlers()
    assert sorted(handlers) == sorted(github_registry.names())
    for spec in github_registry:
        params = inspect.signature(handlers[spec.name]).parameters
        assert spec.argument_names <= set(params), spec.name
        required = {n for n, p in params.items() if p.default is inspect.Parameter.empty}
        assert required <= {a.name for a in spec.required_arguments}, spec.name


def test_write_tools_are_flagged(github_registry):
    writes = {s.name for s in github_registry if s.writes}
    assert writes == {
        "create_issues", "assign_issues", "add_labels", "add_issues_to_milestone", "close_issues"
    }  # fmt: skip


# -------------------------------------------------------------- read tools


def test_search_builds_the_query_and_resolves_me(gh):
    out = tools(gh).search_issues(
        labels="bug, p1", assignee="@me", milestone_number=6, inactive_days=30, sort="reactions"
    )
    params = next(p for m, path, p in gh.calls if path == "/search/issues")
    q = params["q"]
    assert q.startswith(f"repo:{REPO} is:issue is:open")
    assert 'label:"bug"' in q and 'label:"p1"' in q
    assert "assignee:octocat" in q  # "@me" -> the token owner's login
    assert 'milestone:"v2.0"' in q
    assert "updated:<2026-10-06" in q
    assert (params["sort"], params["order"]) == ("reactions", "desc")
    assert out["issue_numbers"] == []
    assert out["message"] == "No issues match."


def test_search_unassigned_returns_numbers_and_titles(gh):
    out = tools(gh).search_issues(labels=["bug"], unassigned=True)
    assert out["issue_numbers"] == [1, 4]
    assert "#1 Login times out on slow networks" in out["message"]


def test_current_milestone_is_the_open_one_due_soonest(gh):
    assert tools(gh).get_milestone()["milestone_number"] == 6
    assert tools(gh).get_milestone(title="V2.1")["milestone_number"] == 7
    with pytest.raises(GitHubError, match="no milestone called"):
        tools(gh).get_milestone(title="v9")


def test_similar_issues_share_title_words_and_exclude_the_issue_itself(gh):
    out = tools(gh).find_similar_issues("#1")
    assert out["similar_issues"] == [2]
    q = next(p["q"] for m, path, p in gh.calls if path == "/search/issues")
    assert re.search(r"\(.+ OR .+\) in:title", q)


def test_similarity_credits_word_variants_but_not_one_shared_word():
    login = "Login times out on slow networks"
    assert _similarity(login, "Login timeout on slow connections") > SIMILARITY_THRESHOLD
    assert _similarity(login, "Login page typo") < SIMILARITY_THRESHOLD
    # In pallets/flask, "flask" is in half the titles, so it is ignored.
    a, b = "flask run fails with FLASK_APP", "flask run should support pep723"
    assert _similarity(a, b, frozenset({"pallets", "flask"})) < _similarity(a, b)


def test_prioritize_puts_p0_first_and_closed_last(gh):
    out = tools(gh).prioritize_issues([5, 3, 1, 4, "#2"])
    assert out["prioritized_issues"] == [4, 2, 3, 1, 5]
    assert out["message"].startswith("Highest priority first: #4 Data loss")


def test_summary_counts_states_labels_and_unassigned(gh):
    summary = tools(gh).summarize_issues([1, 2, 4, 5])["summary"]
    assert "4 issues (3 open, 1 closed)." in summary
    assert "Labels: bug (3)" in summary
    assert "2 of the open issues have nobody assigned." in summary
    assert "Most discussed: #4" in summary


def test_issue_references_in_any_shape(gh):
    t = tools(gh)
    assert t._refs([12, "13", "#14", "https://github.com/a/b/issues/15", 15.0]) == [12, 13, 14, 15]
    with pytest.raises(GitHubError, match="not an issue number"):
        t._refs("the login bug")


# ------------------------------------------------------------- write tools


def test_writes_are_dry_runs_by_default(gh):
    t = tools(gh)
    msg = t.assign_issues([1, 4], ["@me"])["message"]
    assert msg.startswith("[dry run] Would assign @octocat to #1 Login")
    t.add_labels([1], ["p1", "needs-triage"])
    t.add_issues_to_milestone([1], 6)
    t.close_issues([5], reason="stale", comment="Closing after 90 days.")
    assert gh.writes == []


def test_dry_run_still_rejects_issues_that_do_not_exist(gh):
    with pytest.raises(GitHubError):
        tools(gh).assign_issues([1, 999], ["dev"])
    assert gh.writes == []


def test_dry_run_flags_labels_the_repo_does_not_have(gh):
    msg = tools(gh).add_labels([1], ["p1", "needs-triage"])["message"]
    assert "(new label: needs-triage)" in msg


def test_apply_makes_the_changes(gh):
    t = tools(gh, apply=True)
    t.assign_issues([1], ["@me"])
    t.add_issues_to_milestone([1], 6)
    t.close_issues([5], reason="duplicate", comment="Dup of #2")
    assert gh.writes == [
        ("POST", f"/repos/{REPO}/issues/1/assignees", {"assignees": ["octocat"]}),
        ("PATCH", f"/repos/{REPO}/issues/1", {"milestone": 6}),
        ("POST", f"/repos/{REPO}/issues/5/comments", {"body": "Dup of #2"}),
        ("PATCH", f"/repos/{REPO}/issues/5", {"state": "closed", "state_reason": "not_planned"}),
    ]


def test_create_issues_dry_run_returns_placeholders_later_steps_can_use(gh):
    t = tools(gh)
    out = t.create_issues("- Fix flaky login test\n- Update setup docs", labels=["chore"])
    assert out["created_issues"] == ["new-1", "new-2"]
    assert "(labeled chore (new label: chore))" in out["message"]
    msg = t.assign_issues(out["created_issues"], ["dev"])["message"]
    assert "new issue 'Fix flaky login test'" in msg
    assert gh.writes == []


def test_create_issues_apply_posts_one_issue_per_title(gh):
    out = tools(gh, apply=True).create_issues(["A", "B"], milestone_number=7)
    assert out["created_issues"] == [6, 7]
    posted = [body for method, path, body in gh.writes]
    assert posted == [{"title": "A", "milestone": 7}, {"title": "B", "milestone": 7}]


def test_me_without_a_token_explains_what_is_needed():
    with pytest.raises(GitHubError, match="GITHUB_TOKEN"):
        tools(FakeGitHub([], token=None)).who_am_i()


# ------------------------------------------------------------------ client


@pytest.mark.parametrize(
    "status, payload, expected",
    [
        (401, {"message": "Bad credentials"}, "rejected the token"),
        (404, {"message": "Not Found"}, "found nothing at /repos/acme/app"),
        (403, {"message": "API rate limit exceeded"}, "rate limit"),
        (422, {"message": "Validation Failed", "errors": [{"message": "bad label"}]}, "bad label"),
    ],
)
def test_client_turns_http_errors_into_readable_messages(monkeypatch, status, payload, expected):
    def fail(req, timeout, context=None):
        raise urllib.error.HTTPError(
            req.full_url, status, "err", {}, io.BytesIO(json.dumps(payload).encode())
        )

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(GitHubError, match=expected):
        GitHubClient(REPO, token="t").check_access()


def test_client_rejects_malformed_repo_names():
    with pytest.raises(ValueError, match="owner/name"):
        GitHubClient("not-a-repo")


# ------------------------------------------------------------- whole loop


def test_reverse_chain_triages_unassigned_bugs_as_a_dry_run(github_registry, gh):
    """ "Assign the unassigned bugs to me": the issue list comes from a backward chain."""
    step = "Assign the open unassigned bugs to @me."
    llm = ScriptedLLM(
        plan=[step, "Final Answer: Two bugs would be assigned to you."],
        select={step: "assign_issues"},
        arguments={
            "assign_issues": '{"issue_numbers": null, "assignees": ["@me"]}',
            "search_issues": '{"labels": ["bug"], "unassigned": true}',
        },
        providers={("assign_issues", "issue_numbers"): "search_issues"},
    )
    handlers = tools(gh).handlers()
    from reversegpt.executor import Executor

    agent = ReverseGPT(
        llm,
        github_registry,
        StaticRetriever(github_registry),
        Executor(handlers),
        top_k=len(github_registry),
    )
    result = agent.run("Assign the unassigned bugs to me")

    assert [c.api_name for c in result.calls] == ["search_issues", "assign_issues"]
    assert result.calls[1].arguments == {"issue_numbers": "$$PREV[0]", "assignees": ["@me"]}
    assert result.calls[1].message.startswith("[dry run] Would assign @octocat to #1")
    assert gh.writes == []
