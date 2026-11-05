"""GitHub Issues backend for the ``github`` toolkit.

Every tool documented in reversegpt/toolkits/github is a method of
``GitHubIssues``. Tools that change something (create, assign, label, close,
move to a milestone) are dry runs unless the object was built with
``apply=True``: they still check their inputs against the live repository,
then report what they would do without doing it.
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from reversegpt.executor import ExecutionError

log = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.github.com"
MAX_PAGE = 100
# Title trigram similarity needed to call two issues similar. Calibrated on
# pallets/flask titles: unrelated pairs that share one common word scored up
# to ~0.24, duplicates and closely related reports 0.25 and up.
SIMILARITY_THRESHOLD = 0.25


class GitHubError(ExecutionError):
    """A GitHub call failed. The message is shown to the planner and the user."""


def _ssl_context() -> ssl.SSLContext:
    """Verify TLS with certifi's CA bundle when available.

    Python from python.org on macOS does not read the system keychain, so the
    default context fails with CERTIFICATE_VERIFY_FAILED until the user runs
    "Install Certificates.command". certifi avoids that.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


# --------------------------------------------------------------------- client


class Transport(Protocol):
    """What ``GitHubIssues`` needs from a client. Tests pass an in-memory fake."""

    repo: str
    token: str | None

    def request(
        self, method: str, path: str, *, params: dict[str, Any] | None = None, body: Any = None
    ) -> Any: ...


class GitHubClient:
    """Minimal GitHub REST client built on the standard library."""

    def __init__(
        self,
        repo: str,
        token: str | None = None,
        api_url: str = DEFAULT_API_URL,
        timeout: float = 30,
    ):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo or ""):
            raise ValueError(f"repository must look like owner/name, got {repo!r}")
        self.repo = repo
        self.token = token
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self._ssl = _ssl_context()

    def request(
        self, method: str, path: str, *, params: dict[str, Any] | None = None, body: Any = None
    ) -> Any:
        url = self.api_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        req.add_header("User-Agent", "reversegpt")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")

        log.debug("GitHub %s %s", method, url)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raise _http_error(exc, method, path, authenticated=bool(self.token)) from exc
        except urllib.error.URLError as exc:
            raise GitHubError(f"could not reach {self.api_url}: {exc.reason}") from exc
        return json.loads(raw) if raw else None

    def check_access(self) -> dict[str, Any]:
        """Fail early, with a clear message, if the repository can't be reached."""
        return self.request("GET", f"/repos/{self.repo}")


def _http_error(
    exc: urllib.error.HTTPError, method: str, path: str, *, authenticated: bool
) -> GitHubError:
    try:
        payload = json.loads(exc.read() or b"{}")
    except ValueError:
        payload = {}
    detail = str(payload.get("message") or exc.reason)
    for err in payload.get("errors") or []:
        detail += f"; {err.get('message') or err.get('code') if isinstance(err, dict) else err}"

    status = exc.code
    if status == 401:
        return GitHubError("GitHub rejected the token (401 Bad credentials)")
    rate_limited = "rate limit" in detail.lower() or exc.headers.get("X-RateLimit-Remaining") == "0"
    if status in (403, 429) and rate_limited:
        hint = "" if authenticated else " Set GITHUB_TOKEN to raise the limit."
        return GitHubError(f"GitHub rate limit reached.{hint}")
    if status == 404:
        return GitHubError(
            f"GitHub found nothing at {path}. Check the repository name"
            + (" and that the token can see it." if authenticated else ", or set GITHUB_TOKEN.")
        )
    if status == 403 and not authenticated:
        return GitHubError(f"{method} {path} needs a GitHub token (set GITHUB_TOKEN)")
    return GitHubError(f"GitHub {method} {path} failed ({status}): {detail}")


# -------------------------------------------------------------------- helpers

_ISSUE_URL = re.compile(r"/issues/(\d+)")
_WORD = re.compile(r"[a-z][a-z0-9]+")  # also splits snake_case identifiers
_STOPWORDS = frozenset(
    """
    about above after again all also and any are because been before being but can
    cannot could did does doesn done each error errors for from get gets getting had
    has have how into isn issue issues its just like make makes more most new not
    only other our out over should some such than that the their them then there
    these they this too under use used using via was wasn way were what when where
    which while who why will with without won work works working would you your
    bug bugs feature request add adds support fix fixes fails failed failing problem
    """.split()  # noqa: SIM905  (a word block is easier to edit than a 100-line literal)
)

# Label patterns, most urgent first. The first rule a label matches gives its weight.
_PRIORITY_RULES = (
    (100, re.compile(r"\b(p0|critical|blocker|urgent|security|sev[-: ]?[01])\b")),
    (60, re.compile(r"\b(p1|high|regression|sev[-: ]?2)\b")),
    (30, re.compile(r"\b(bug|crash|data[- ]loss)\b")),
    (15, re.compile(r"\b(p2|medium|sev[-: ]?3)\b")),
    (
        0,
        re.compile(
            r"\b(p3|p4|low|minor|trivial|enhancement|feature|docs?|documentation|question|chore)\b"
        ),
    ),
)
_UNLABELED_WEIGHT = 10


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out: list[Any] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [value]


def _strings(value: Any, *, split: bool = False) -> list[str]:
    """Normalise a model-supplied value to a list of non-empty strings.

    ``split`` also splits "bug, p1" into two items, which suits labels and
    logins but not issue titles.
    """
    out: list[str] = []
    for item in _flatten(value):
        if item is None:
            continue
        text = str(item).strip()
        parts = text.split(",") if split else [text]
        out.extend(p.strip() for p in parts if p.strip())
    return list(dict.fromkeys(out))


def _titles(value: Any) -> list[str]:
    """Issue titles: a list, or one block of text with one title per line."""
    titles: list[str] = []
    for item in _strings(value):
        for line in item.splitlines():
            line = line.strip().lstrip("-*•").strip()
            if line:
                titles.append(line)
    return titles


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "y", "1")
    return bool(value)


def _int(
    value: Any, name: str, *, default: int | None = None, high: int | None = None
) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    try:
        number = int(str(value).strip().lstrip("#"))
    except ValueError:
        raise GitHubError(f"{name} must be a number, got {value!r}") from None
    number = max(number, 1)
    return min(number, high) if high else number


def _parse_time(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _short(text: str, limit: int = 60) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _words(text: str, ignore: frozenset[str] = frozenset()) -> list[str]:
    """Meaningful words in order, without repeats. ``ignore`` adds per-repo stopwords."""
    words = _WORD.findall(text.lower())
    keep = (w for w in words if len(w) >= 3 and w not in _STOPWORDS and w not in ignore)
    return list(dict.fromkeys(keep))


def _keywords(text: str, ignore: frozenset[str] = frozenset(), limit: int = 5) -> list[str]:
    """The most distinctive words of a title. Longer words tend to be more specific."""
    return sorted(_words(text, ignore), key=len, reverse=True)[:limit]


def _trigrams(text: str, ignore: frozenset[str] = frozenset()) -> set[str]:
    grams: set[str] = set()
    for word in _words(text, ignore):
        padded = f"  {word} "
        grams.update(padded[i : i + 3] for i in range(len(padded) - 2))
    return grams


def _similarity(a: str, b: str, ignore: frozenset[str] = frozenset()) -> float:
    """Trigram similarity of two titles, 0..1, as in PostgreSQL's pg_trgm.

    Unlike exact word overlap it credits variants such as "times out" and
    "timeout", or "crash" and "crashes", without a stemmer.
    """
    ta, tb = _trigrams(a, ignore), _trigrams(b, ignore)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def _label_names(issue: dict[str, Any]) -> list[str]:
    return [lab["name"] if isinstance(lab, dict) else str(lab) for lab in issue.get("labels", [])]


# ---------------------------------------------------------------------- tools


class GitHubIssues:
    """The operations behind the GitHub toolkit, bound to one repository."""

    def __init__(
        self,
        client: Transport,
        *,
        apply: bool = False,
        now: Callable[[], datetime] | None = None,
    ):
        self.client = client
        self.apply = apply
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._issues: dict[int, dict[str, Any]] = {}
        self._milestones: dict[int, dict[str, Any]] = {}
        self._repo_labels: set[str] | None = None
        self._login: str | None = None
        self._drafts: dict[str, str] = {}  # dry-run placeholder -> title
        # The repo's own name ("flask" in pallets/flask) is in half its titles,
        # so it says nothing about whether two issues are alike.
        self._repo_words = frozenset(_WORD.findall(client.repo.lower()))

    @property
    def repo(self) -> str:
        return self.client.repo

    def handlers(self) -> dict[str, Callable[..., dict[str, Any]]]:
        return {
            "who_am_i": self.who_am_i,
            "search_issues": self.search_issues,
            "get_milestone": self.get_milestone,
            "find_similar_issues": self.find_similar_issues,
            "prioritize_issues": self.prioritize_issues,
            "summarize_issues": self.summarize_issues,
            "create_issues": self.create_issues,
            "assign_issues": self.assign_issues,
            "add_labels": self.add_labels,
            "add_issues_to_milestone": self.add_issues_to_milestone,
            "close_issues": self.close_issues,
        }

    # ----------------------------------------------------------- read tools

    def who_am_i(self) -> dict[str, Any]:
        login = self._me()
        return {"login": login, "message": f"The token belongs to @{login}."}

    def search_issues(
        self,
        state: str | None = "open",
        labels: Any = None,
        assignee: Any = None,
        unassigned: Any = False,
        author: Any = None,
        milestone_number: Any = None,
        no_milestone: Any = False,
        text: str | None = None,
        inactive_days: Any = None,
        sort: str | None = None,
        limit: Any = 30,
    ) -> dict[str, Any]:
        terms = [f"repo:{self.repo}", "is:issue"]

        state = str(state or "open").strip().lower()
        if state in ("open", "closed"):
            terms.append(f"is:{state}")
        elif state != "all":
            raise GitHubError(f'state must be "open", "closed" or "all", got {state!r}')

        terms += [f'label:"{name}"' for name in _strings(labels, split=True)]

        people = _strings(assignee, split=True)
        if _truthy(unassigned) or any(p.lower() in ("none", "nobody") for p in people):
            terms.append("no:assignee")
        else:
            terms += [f"assignee:{self._login_of(p)}" for p in people]
        terms += [f"author:{self._login_of(p)}" for p in _strings(author, split=True)]

        if not (milestone_number is None or milestone_number == ""):
            terms.append(f'milestone:"{self._milestone(milestone_number)["title"]}"')
        elif _truthy(no_milestone):
            terms.append("no:milestone")

        days = _int(inactive_days, "inactive_days")
        if days:
            cutoff = (self._now() - timedelta(days=days)).date().isoformat()
            terms.append(f"updated:<{cutoff}")

        if text and str(text).strip():
            terms.append(str(text).strip())

        limit = _int(limit, "limit", default=30, high=MAX_PAGE)
        params: dict[str, Any] = {"q": " ".join(terms), "per_page": limit}
        if sort and str(sort).strip().lower() in ("created", "updated", "comments", "reactions"):
            params.update(sort=str(sort).strip().lower(), order="desc")

        data = self.client.request("GET", "/search/issues", params=params)
        items = [i for i in data.get("items", []) if "pull_request" not in i][:limit]
        self._remember(items)
        numbers = [i["number"] for i in items]
        if not numbers:
            return {"issue_numbers": [], "message": "No issues match."}

        total = data.get("total_count", len(numbers))
        head = f"Found {total} matching issues"
        if total > len(numbers):
            head += f" (showing {len(numbers)})"
        return {"issue_numbers": numbers, "message": f"{head}: {self._list(numbers)}."}

    def get_milestone(self, title: str | None = None) -> dict[str, Any]:
        wanted = str(title or "").strip().lower()
        milestones = self.client.request(
            "GET",
            f"/repos/{self.repo}/milestones",
            params={"state": "all" if wanted else "open", "per_page": MAX_PAGE},
        )
        for m in milestones:
            self._milestones[m["number"]] = m

        if wanted:
            milestones.sort(key=lambda m: m.get("state") != "open")  # prefer open ones
            match = next((m for m in milestones if m["title"].lower() == wanted), None) or next(
                (m for m in milestones if wanted in m["title"].lower()), None
            )
            if match is None:
                names = ", ".join(m["title"] for m in milestones[:10]) or "none"
                raise GitHubError(f"no milestone called {title!r} (milestones: {names})")
        else:
            if not milestones:
                raise GitHubError("the repository has no open milestones")
            dated = sorted((m for m in milestones if m.get("due_on")), key=lambda m: m["due_on"])
            match = dated[0] if dated else max(milestones, key=lambda m: m["number"])

        due = f", due {match['due_on'][:10]}" if match.get("due_on") else ""
        return {
            "milestone_number": match["number"],
            "message": (
                f"Milestone '{match['title']}' (#{match['number']}, "
                f"{match.get('open_issues', 0)} open issues{due})."
            ),
        }

    def find_similar_issues(self, issue_number: Any, limit: Any = 5) -> dict[str, Any]:
        number = self._single(issue_number)
        issue = self._issue(number)
        ignore = self._repo_words
        words = _keywords(issue["title"], ignore) or _keywords(issue.get("body") or "", ignore)
        if not words:
            return {
                "similar_issues": [],
                "message": f"#{number} has no distinctive words to search for.",
            }

        query = f"repo:{self.repo} is:issue ({' OR '.join(words)}) in:title"
        data = self.client.request("GET", "/search/issues", params={"q": query, "per_page": 50})

        scored = [
            (_similarity(issue["title"], item["title"], ignore), item)
            for item in data.get("items", [])
            if item["number"] != number and "pull_request" not in item
        ]
        scored = [s for s in scored if s[0] >= SIMILARITY_THRESHOLD]
        scored.sort(key=lambda s: (-s[0], -s[1]["number"]))

        top = [item for _, item in scored[: _int(limit, "limit", default=5, high=20)]]
        self._remember(top)
        numbers = [item["number"] for item in top]
        title = _short(issue["title"])
        if not numbers:
            return {
                "similar_issues": [],
                "message": f"No issues look similar to #{number} {title}.",
            }
        found = "; ".join(
            f"#{item['number']} {_short(item['title'], 50)} ({item['state']})" for item in top
        )
        return {
            "similar_issues": numbers,
            "message": f"Issues similar to #{number} {title}: {found}.",
        }

    def prioritize_issues(self, issue_numbers: Any) -> dict[str, Any]:
        numbers = self._existing(self._refs(issue_numbers))
        if not numbers:
            raise GitHubError("there are no existing issues to prioritize")
        scores = {n: self._score(self._issues[n]) for n in numbers}
        ranked = sorted(numbers, key=lambda n: (-scores[n][0], n))
        top = "; ".join(
            f"#{n} {_short(self._issues[n]['title'], 40)} ({scores[n][1]})" for n in ranked[:5]
        )
        more = f" (+{len(ranked) - 5} more)" if len(ranked) > 5 else ""
        return {"prioritized_issues": ranked, "message": f"Highest priority first: {top}{more}."}

    def summarize_issues(self, issue_numbers: Any) -> dict[str, Any]:
        numbers = self._existing(self._refs(issue_numbers))
        if not numbers:
            raise GitHubError("there are no existing issues to summarize")
        issues = [self._issues[n] for n in numbers]
        still_open = [i for i in issues if i.get("state") == "open"]
        now = self._now()

        lines = [
            f"{len(issues)} issues ({len(still_open)} open, {len(issues) - len(still_open)} closed)."
        ]
        labels = Counter(name for i in issues for name in _label_names(i))
        if labels:
            common = ", ".join(f"{name} ({count})" for name, count in labels.most_common(5))
            lines.append(f"Labels: {common}.")
        if still_open:
            unassigned = sum(1 for i in still_open if not i.get("assignees"))
            lines.append(f"{unassigned} of the open issues have nobody assigned.")
            oldest = min(still_open, key=lambda i: i["created_at"])
            age = (now - _parse_time(oldest["created_at"])).days
            lines.append(
                f"Oldest open: #{oldest['number']} {_short(oldest['title'])} ({age} days)."
            )
        busiest = max(issues, key=lambda i: i.get("comments", 0))
        if busiest.get("comments"):
            lines.append(
                f"Most discussed: #{busiest['number']} {_short(busiest['title'])} "
                f"({busiest['comments']} comments)."
            )
        lines.append(f"Issues: {self._list(numbers, limit=10)}.")
        summary = "\n".join(lines)
        return {"summary": summary, "message": summary}

    # ---------------------------------------------------------- write tools

    def create_issues(
        self,
        titles: Any,
        body: str | None = None,
        labels: Any = None,
        assignees: Any = None,
        milestone_number: Any = None,
    ) -> dict[str, Any]:
        titles = _titles(titles)
        if not titles:
            raise GitHubError("create_issues needs at least one title")
        label_names = _strings(labels, split=True)
        logins = [self._login_of(a) for a in _strings(assignees, split=True)]
        milestone = (
            None
            if milestone_number is None or milestone_number == ""
            else self._milestone(milestone_number)
        )

        details = []
        if label_names:
            details.append(f"labeled {', '.join(label_names)}{self._new_label_note(label_names)}")
        if logins:
            details.append("assigned to " + ", ".join(f"@{x}" for x in logins))
        if milestone:
            details.append(f"in milestone '{milestone['title']}'")
        extra = f" ({'; '.join(details)})" if details else ""

        if not self.apply:
            refs = []
            for title in titles:
                ref = f"new-{len(self._drafts) + 1}"
                self._drafts[ref] = title
                refs.append(ref)
            listing = "; ".join(f"'{_short(t)}'" for t in titles)
            return {
                "created_issues": refs,
                "message": f"[dry run] Would create {len(titles)} issue(s){extra}: {listing}.",
            }

        fields = {
            "body": body or None,
            "labels": label_names or None,
            "assignees": logins or None,
            "milestone": milestone["number"] if milestone else None,
        }
        fields = {k: v for k, v in fields.items() if v is not None}
        created = []
        for title in titles:
            issue = self.client.request(
                "POST", f"/repos/{self.repo}/issues", body={"title": title, **fields}
            )
            self._remember([issue])
            created.append(issue["number"])
        return {
            "created_issues": created,
            "message": f"Created {len(created)} issue(s){extra}: {self._list(created, limit=10)}.",
        }

    def assign_issues(self, issue_numbers: Any, assignees: Any) -> dict[str, Any]:
        refs = self._refs(issue_numbers)
        numbers = self._existing(refs)
        logins = [self._login_of(a) for a in _strings(assignees, split=True)]
        if not logins:
            raise GitHubError("assign_issues needs at least one assignee")
        who = ", ".join(f"@{x}" for x in logins)

        if not self.apply:
            return {"message": f"[dry run] Would assign {who} to {self._list(refs, limit=8)}."}
        for n in numbers:
            self.client.request(
                "POST", f"/repos/{self.repo}/issues/{n}/assignees", body={"assignees": logins}
            )
        return {"message": f"Assigned {who} to {self._list(refs, limit=8)}."}

    def add_labels(self, issue_numbers: Any, labels: Any) -> dict[str, Any]:
        refs = self._refs(issue_numbers)
        numbers = self._existing(refs)
        names = _strings(labels, split=True)
        if not names:
            raise GitHubError("add_labels needs at least one label")
        what = ", ".join(names) + self._new_label_note(names)

        if not self.apply:
            return {"message": f"[dry run] Would label {self._list(refs, limit=8)} with {what}."}
        for n in numbers:
            self.client.request(
                "POST", f"/repos/{self.repo}/issues/{n}/labels", body={"labels": names}
            )
        return {"message": f"Labeled {self._list(refs, limit=8)} with {what}."}

    def add_issues_to_milestone(self, issue_numbers: Any, milestone_number: Any) -> dict[str, Any]:
        refs = self._refs(issue_numbers)
        numbers = self._existing(refs)
        milestone = self._milestone(milestone_number)
        where = f"milestone '{milestone['title']}'"

        if not self.apply:
            return {"message": f"[dry run] Would add {self._list(refs, limit=8)} to {where}."}
        for n in numbers:
            self.client.request(
                "PATCH", f"/repos/{self.repo}/issues/{n}", body={"milestone": milestone["number"]}
            )
        return {"message": f"Added {self._list(refs, limit=8)} to {where}."}

    def close_issues(
        self, issue_numbers: Any, reason: str | None = None, comment: str | None = None
    ) -> dict[str, Any]:
        refs = self._refs(issue_numbers)
        numbers = self._existing(refs)
        words = str(reason or "").lower()
        not_planned = any(w in words for w in ("not", "stale", "dup", "won", "invalid", "inactiv"))
        state_reason = "not_planned" if not_planned else "completed"
        comment = (comment or "").strip()
        note = f", commenting '{_short(comment, 80)}'" if comment else ""
        how = state_reason.replace("_", " ")

        if not self.apply:
            return {"message": f"[dry run] Would close {self._list(refs, limit=8)} as {how}{note}."}
        for n in numbers:
            if comment:
                self.client.request(
                    "POST", f"/repos/{self.repo}/issues/{n}/comments", body={"body": comment}
                )
            self.client.request(
                "PATCH",
                f"/repos/{self.repo}/issues/{n}",
                body={"state": "closed", "state_reason": state_reason},
            )
        return {"message": f"Closed {self._list(refs, limit=8)} as {how}{note}."}

    # -------------------------------------------------------------- internals

    def _me(self) -> str:
        if self._login is None:
            if not self.client.token:
                raise GitHubError("knowing who 'me' is needs a GitHub token (set GITHUB_TOKEN)")
            self._login = self.client.request("GET", "/user")["login"]
        return self._login

    def _login_of(self, name: str) -> str:
        name = str(name).strip().lstrip("@")
        if name.lower() in ("", "me", "myself", "i", "self"):
            return self._me()
        return name

    def _refs(self, value: Any) -> list[int | str]:
        """Issue numbers from whatever the model passed: 12, "12", "#12", a URL, a list."""
        refs: list[int | str] = []
        for item in _flatten(value):
            if isinstance(item, bool):
                raise GitHubError(f"{item!r} is not an issue number")
            if isinstance(item, int) or (isinstance(item, float) and item.is_integer()):
                refs.append(int(item))
                continue
            text = str(item).strip()
            if text in self._drafts:
                refs.append(text)
                continue
            match = _ISSUE_URL.search(text)
            digits = match.group(1) if match else text.lstrip("#")
            if not digits.isdigit():
                raise GitHubError(f"{item!r} is not an issue number")
            refs.append(int(digits))
        return list(dict.fromkeys(refs))

    def _single(self, value: Any) -> int:
        numbers = [r for r in self._refs(value) if isinstance(r, int)]
        if not numbers:
            raise GitHubError(f"{value!r} is not an issue number")
        return numbers[0]

    def _issue(self, number: int) -> dict[str, Any]:
        if number not in self._issues:
            self._issues[number] = self.client.request("GET", f"/repos/{self.repo}/issues/{number}")
        return self._issues[number]

    def _existing(self, refs: list[int | str]) -> list[int]:
        """Fetch every real issue first, so a wrong number fails before anything changes."""
        numbers = [r for r in refs if isinstance(r, int)]
        for n in numbers:
            self._issue(n)
        return numbers

    def _remember(self, items: Iterable[dict[str, Any]]) -> None:
        for item in items:
            self._issues[item["number"]] = item

    def _milestone(self, number: Any) -> dict[str, Any]:
        n = _int(number, "milestone_number")
        if n not in self._milestones:
            self._milestones[n] = self.client.request("GET", f"/repos/{self.repo}/milestones/{n}")
        return self._milestones[n]

    def _new_label_note(self, names: list[str]) -> str:
        if self._repo_labels is None:
            existing = self.client.request(
                "GET", f"/repos/{self.repo}/labels", params={"per_page": MAX_PAGE}
            )
            self._repo_labels = {lab["name"].lower() for lab in existing}
        new = [n for n in names if n.lower() not in self._repo_labels]
        return f" (new label: {', '.join(new)})" if new else ""

    def _describe(self, ref: int | str) -> str:
        if isinstance(ref, str):
            return f"new issue '{_short(self._drafts[ref], 40)}'"
        issue = self._issues.get(ref)
        return f"#{ref} {_short(issue['title'], 50)}" if issue else f"#{ref}"

    def _list(self, refs: list[int] | list[int | str], limit: int = 5) -> str:
        shown = "; ".join(self._describe(r) for r in refs[:limit])
        return shown + (f" (+{len(refs) - limit} more)" if len(refs) > limit else "")

    def _score(self, issue: dict[str, Any]) -> tuple[float, str]:
        """Urgency from labels, plus engagement and age. Returns (score, reason)."""
        best: tuple[int, str] | None = None
        for name in _label_names(issue):
            for weight, pattern in _PRIORITY_RULES:
                if pattern.search(name.lower()):
                    if best is None or weight > best[0]:
                        best = (weight, name)
                    break
        label_weight = best[0] if best else _UNLABELED_WEIGHT
        reactions = (issue.get("reactions") or {}).get("total_count", 0)
        comments = issue.get("comments", 0)
        age = (self._now() - _parse_time(issue["created_at"])).days
        closed = issue.get("state") == "closed"

        score = label_weight + min(reactions, 50) + 0.5 * min(comments, 40) + min(age / 30, 10)
        if closed:
            score -= 200

        reasons = []
        if best:
            reasons.append(best[1])
        if reactions:
            reasons.append(f"{reactions} reactions")
        if comments:
            reasons.append(f"{comments} comments")
        if closed:
            reasons.append("closed")
        return score, ", ".join(reasons) or f"open {age} days"
