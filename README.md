# ReverseGPT

**Triage your GitHub issues in plain English.**

```bash
reversegpt run --repo your-org/your-app \
  "Assign the unassigned p1 bugs to me and add them to the current milestone"
```

ReverseGPT turns one sentence into the right sequence of GitHub API calls. It finds the
issues, works out who "me" is and which milestone is current, shows you exactly what it is
about to change, and changes nothing until you add `--apply`.

## The problem

Every active repository needs triage: label new reports, spot duplicates, decide what to fix
first, give each bug an owner, plan the next milestone, close what has gone stale. None of it
is hard. All of it is tedious: GitHub's search syntax, a dozen browser tabs, issue numbers
copied by hand from one page to another.

So triage slips. Backlogs fill with duplicates and stale questions, urgent bugs sit with no
owner, and maintainers lose time to bookkeeping instead of fixing things.

## What you can ask

| Job | You type | What runs |
| --- | --- | --- |
| Weekly triage | *Assign the unassigned p1 bugs to me and add them to the current milestone* | search, assign, find milestone, move |
| Duplicate check | *Is #482 a duplicate? Summarize anything similar* | find similar issues, summarize |
| What to fix first | *Prioritize the open p0 and p1 bugs* | search, rank by urgency |
| Stand-up digest | *Summarize the open bugs in the v2.1 milestone* | find milestone, search, summarize |
| Stale cleanup | *Close questions with no activity for 90 days and say why* | search by inactivity, close with a comment |
| Notes to issues | *Turn these meeting notes into issues labeled from-standup* | create one issue per action item |

Piping works too, so meeting notes can go straight in:

```bash
(echo "Turn these notes into issues labeled from-standup:"; pbpaste) | reversegpt run --repo you/app -
```

## See it work

A dry run against [psf/requests](https://github.com/psf/requests) for
*"Prioritize the open bugs, summarize them, and label the top one needs-triage"*
(real data from the GitHub API, lightly trimmed):

```text
[0] search_issues(state="open", labels=["bug"], limit=6)
    Found 8 matching issues (showing 6): #6102 HTTPDigestAuth fails on non-latin credentials;
    #4965 Accessing response.content twice removes forget...; #3829 Session.verify=False ignored
    when REQUESTS_CA_B...; #3296 no_proxy env ignored on 302 redirect; ... (+1 more).
[1] prioritize_issues(issue_numbers="$$PREV[0]")
    Highest priority first: #3829 Session.verify=False ignored when REQ... (Bug, 25 reactions,
    14 comments); #6102 HTTPDigestAuth fails on non-latin cre... (Bug, 9 comments); ...
[2] summarize_issues(issue_numbers="$$PREV[1]")
    6 issues (6 open, 0 closed).
    Labels: Bug (6), Planned (2), Breaking API Change (1).
    6 of the open issues have nobody assigned.
    Oldest open: #2392 requests.exceptions.ReadTimeout is not raised when respon... (4331 days).
    Most discussed: #3829 Session.verify=False ignored when REQUESTS_CA_BUNDLE envi... (14 comments).
[3] add_labels(issue_numbers=[3829], labels=["needs-triage"])
    [dry run] Would label #3829 Session.verify=False ignored when REQUESTS_CA_B... with
    needs-triage (new label: needs-triage).

Dry run: 1 change(s) previewed, nothing was modified. Re-run with --apply to make them.
```

Note the first two lines: the request never said to *search* for anything. The ranking tool
needed issue numbers, so ReverseGPT worked backwards to the tool that produces them. That is
the Reverse Chain, explained [below](#how-it-works).

## Safe by default

An assistant that can edit your tracker has to be trustworthy, so:

- **Every change is a dry run unless you pass `--apply`.** The dry run still checks against
  the live repository: issues must exist, milestones must exist, and labels the repo doesn't
  have yet are flagged (catching typos like `bgu`).
- **`--apply` needs a token**, and can only do what that token is allowed to do. A
  fine-grained token limited to one repository and *Issues: read and write* is enough.
- **Every run is recorded.** The exact API calls are saved to `output/output.json` and the
  full trace, including prompts, to `logs/run.log`.
- **Reading needs no token at all** for public repositories, so you can try it on any open
  source project without granting access to anything.

## Quick start

Python 3.10 or newer.

```bash
pip install git+https://github.com/Lazypilottt/ReverseGPT.git

export OPENAI_API_KEY=sk-...
export GITHUB_TOKEN=github_pat_...    # or: export GITHUB_TOKEN=$(gh auth token)

reversegpt run --repo your-org/your-app "Summarize the open bugs"
reversegpt run --repo your-org/your-app "Close questions with no activity for 90 days" --apply
```

Without `GITHUB_TOKEN` it still reads public repositories (GitHub allows 60 requests an hour
that way). Any OpenAI-compatible model server works too, such as a local
[Ollama](https://ollama.com): set `base_url` in `config.ini`.

Embeddings for tool search run locally with `sentence-transformers`, which installs PyTorch.
On Linux without a GPU, install the CPU build first to skip a multi-gigabyte CUDA download:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## How it works

![Architecture](images/architecture.jpg)

Most LLM agents ask one model call to plan a whole workflow, pick tools and fill every
argument at once. That breaks easily once tools depend on each other. ReverseGPT combines
two ideas from research on tool use:

1. **A planner** ([RestGPT](https://arxiv.org/abs/2306.06624)-style) writes one step at a
   time in plain English, looking at what earlier steps returned.
2. **Reverse Chain** ([paper](https://arxiv.org/abs/2310.04474)) carries out each step
   backwards. It picks the API that *finishes* the step, fills the arguments it can, and for
   each required argument that is still missing, finds the API that *produces* it and
   repeats.

Take *"Assign the unassigned bugs to me"*:

```text
assign_issues(issue_numbers=?, assignees=["@me"])    the API that finishes the job
  issue_numbers is missing: which API returns issue numbers?
  search_issues(labels=["bug"], unassigned=true)      runs first, returns $$PREV[0]
assign_issues(issue_numbers=$$PREV[0], assignees=["@me"])
```

Each model call makes one small decision: which tool, which arguments, or which tool provides
a missing value. That keeps prompts short, makes failures easy to see in the log, and means
new tools need no new workflow code, just a JSON description. A FAISS index over the tool
descriptions shortlists candidates at each decision, so the model only ever sees a handful of
tools.

### Under the hood of the GitHub tools

- **Duplicates.** `find_similar_issues` searches the repository using the most distinctive
  words of the title, then ranks candidates by character-trigram similarity (the measure
  PostgreSQL's `pg_trgm` uses). That credits variants like *"times out"* and *"timeout"*,
  and ignores the repository's own name, which appears in half its titles.
- **Priority.** `prioritize_issues` ranks by labels first (p0, critical, blocker and security,
  then p1, high and regression, then bug, then p2, then low, docs and enhancement), then
  reactions, discussion and age. Closed issues sink to the bottom. Every rank comes with its
  reason.
- **Digests.** `summarize_issues` reports open and closed counts, the most common labels,
  unassigned work, the oldest open issue and the most discussed one.

## Toolkits

A toolkit is a folder of JSON tool descriptions plus the code that runs them.

| Toolkit | What it drives | Tools |
| --- | --- | --- |
| `github` | Your real GitHub issues | `search_issues`, `get_milestone`, `find_similar_issues`, `prioritize_issues`, `summarize_issues`, `who_am_i`, and the write tools `create_issues`, `assign_issues`, `add_labels`, `add_issues_to_milestone`, `close_issues` |
| `devrev` | An offline mock of the DevRev API, the project's original demo | Work items, sprints and customers. Needs no account. |

```bash
reversegpt tools --repo your-org/your-app   # list the GitHub tools (* = changes data)
reversegpt tools --toolkit devrev           # the offline demo
reversegpt run --toolkit devrev "Prioritize my P0 issues and add them to the current sprint"
```

### Bring your own API

The engine knows nothing about GitHub. Point it at any API by describing each endpoint:

```json
{
  "name": "get_similar_work_items",
  "description": "Returns a list of work items that are similar to the given work item.",
  "arguments": [
    { "name": "work_id", "type": "string", "required": true,
      "description": "The ID of the work item to find similar items for." }
  ],
  "returns": [
    { "name": "similar_work_items", "type": "array of strings",
      "description": "IDs of work items similar to the given one." }
  ],
  "examples": ["Find issues similar to don:core:dvrv-us-1:devo/0:issue/1"]
}
```

Set `docs_dir` in `config.ini` to your folder, add `"writes": true` to tools that change data,
and optionally put worked examples for the planner in `planner_examples.txt` in the same
folder. Then give the agent a function per tool. Each one receives the arguments as keyword
arguments (dots become underscores) and returns a dict keyed by the `returns` names, plus an
optional `message`:

```python
from reversegpt import ReverseGPT, load_settings
from reversegpt.executor import Executor

def work_list(limit=50, **filters):
    ids = my_client.list_work(limit=limit, **filters)
    return {"work_ids": ids, "message": f"Found {len(ids)} work items."}

agent = ReverseGPT.from_settings(load_settings("config.ini"))
agent.executor = Executor(handlers={"work_list": work_list, ...})
result = agent.run("Summarize high severity tickets from customer Acme")
print(result.tool_calls())
```

Tools without a handler return placeholder outputs, so you can try planning before writing
any integration code.

## Configure

Copy `config.example.ini` to `config.ini`. Every key is optional, and command line flags
override it.

| Section | Key | Default |
| --- | --- | --- |
| `[tools]` | `toolkit` | `devrev` (`--repo` switches to `github`) |
| `[github]` | `repo` | none (set it here or with `--repo owner/name`) |
| `[github]` | `token` | `GITHUB_TOKEN` or `GH_TOKEN` from the environment |
| `[github]` | `api_url` | `https://api.github.com` (change for GitHub Enterprise Server) |
| `[llm]` | `model` | `gpt-4o-mini` |
| `[llm]` | `base_url` | OpenAI. Any OpenAI-compatible server works. |
| `[retriever]` | `top_k` | `5` candidate tools per decision |
| `[agent]` | `max_plan_steps` / `max_chain_depth` | `8` / `3` |

Other `run` options: `--json` prints the raw tool calls, `-o file.json` saves them elsewhere,
`-v` shows every prompt and reply, and `-` reads the request from stdin.

## Development

```bash
git clone https://github.com/Lazypilottt/ReverseGPT.git && cd ReverseGPT
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

The tests run offline: a scripted model stands in for the LLM and an in-memory fake stands in
for GitHub. The tests in `tests/test_retriever.py` that check real embedding quality download
a model and are skipped in CI.

## License

MIT. See [LICENSE](LICENSE).
