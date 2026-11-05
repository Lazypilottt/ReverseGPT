"""RestGPT-style planner: decides the next natural-language step to take."""

from __future__ import annotations

import re
from dataclasses import dataclass

from reversegpt.llm import ChatModel, complete
from reversegpt.registry import ApiRegistry

EXAMPLES = """\
Example 1
User query: Prioritize my P0 issues and add them to the current sprint.
Plan step 1: Get the ID of the current user.
API response: who_am_i returned user_id = $$PREV[0].
Plan step 2: Get the P0 issues owned by the current user ($$PREV[0]).
API response: work_list returned work_ids = $$PREV[1].
Plan step 3: Prioritize the P0 issues ($$PREV[1]).
API response: prioritize_objects returned prioritized_objects = $$PREV[2].
Plan step 4: Get the ID of the current sprint.
API response: get_sprint_id returned sprint_id = $$PREV[3].
Plan step 5: Add the prioritized issues ($$PREV[2]) to the current sprint ($$PREV[3]).
API response: add_work_items_to_sprint ran successfully.
Thought: The P0 issues are prioritized and in the current sprint, so the query is done.
Final Answer: Your P0 issues were prioritized and added to the current sprint.

Example 2
User query: Summarize issues similar to don:core:dvrv-us-1:devo/0:issue/1.
Plan step 1: Get the work items similar to don:core:dvrv-us-1:devo/0:issue/1.
API response: get_similar_work_items returned similar_work_items = $$PREV[0].
Plan step 2: Summarize the similar work items ($$PREV[0]).
API response: summarize_objects returned summary = $$PREV[1].
Thought: The similar issues have been summarized, so the query is done.
Final Answer: Here is a summary of the issues similar to don:core:dvrv-us-1:devo/0:issue/1.

Example 3
User query: Summarize high severity tickets from the customer UltimateCustomer.
Plan step 1: Find the ID of the customer named UltimateCustomer.
API response: search_object_by_name returned id = $$PREV[0].
Plan step 2: Get high severity tickets from the customer $$PREV[0].
API response: work_list returned work_ids = $$PREV[1].
Plan step 3: Summarize the tickets ($$PREV[1]).
API response: summarize_objects returned summary = $$PREV[2].
Thought: The high severity tickets have been summarized, so the query is done.
Final Answer: Here is a summary of the high severity tickets from UltimateCustomer.
"""

PROMPT = """\
You plan how to answer a user's request with the tools listed below.
You do not call tools yourself. You write one plan step at a time in plain
English; another component picks the tool for that step, runs it, and reports
back with an "API response" line.

Rules:
- Each step should be something a single tool call can do.
- Put filters, sorting and search conditions in the same step as the fetch.
- Be specific. Instead of pronouns, refer to earlier results by their
  $$PREV[i] reference or by the literal value from the query.
- Do not repeat a step that already succeeded.
- If the last API response does not complete the step, write "Continue" to
  retry the same step with a different tool.
- When the request is fully handled, write a "Thought:" line followed by a
  "Final Answer:" line.
- If the request cannot be handled with these tools at all, reply with [] and
  nothing else.

Available tools:
{tools}

{examples}
Now the real request. Reply with only the next plan step, or with the
Thought / Final Answer lines if you are done.

User query: {query}
{scratchpad}Plan step {step}:"""

_STEP_PREFIX = re.compile(r"^\s*Plan step \d+\s*:\s*", re.IGNORECASE)


@dataclass
class PlanStep:
    text: str = ""
    final: bool = False
    answer: str | None = None
    unrelated: bool = False
    retry: bool = False


class Planner:
    def __init__(self, llm: ChatModel, registry: ApiRegistry, examples: str = EXAMPLES):
        self.llm = llm
        self.examples = examples
        self.tools = "\n".join(f"- {s.name}: {s.description}" for s in registry)

    def build_prompt(self, query: str, history: list[tuple[str, str]]) -> str:
        scratchpad = "".join(
            f"Plan step {i}: {step}\nAPI response: {response}\n"
            for i, (step, response) in enumerate(history, start=1)
        )
        return PROMPT.format(
            tools=self.tools,
            examples=self.examples,
            query=query,
            scratchpad=scratchpad,
            step=len(history) + 1,
        )

    def next_step(self, query: str, history: list[tuple[str, str]]) -> PlanStep:
        prompt = self.build_prompt(query, history)
        reply = complete(self.llm, prompt, stop=["\nAPI response"])
        return self.parse(reply)

    @staticmethod
    def parse(reply: str) -> PlanStep:
        text = _STEP_PREFIX.sub("", reply.strip()).strip()

        if text in ("[]", "") or text.startswith("[]"):
            return PlanStep(unrelated=True)

        match = re.search(r"Final Answer\s*:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        if match:
            return PlanStep(final=True, answer=match.group(1).strip())

        # Keep only the step itself, skipping any "Thought:" lines around it.
        lines = [
            _STEP_PREFIX.sub("", line).strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().lower().startswith("thought:")
        ]
        if not lines:
            return PlanStep(final=True)
        step = lines[0]
        if step.lower().rstrip(".") == "continue":
            return PlanStep(retry=True)
        return PlanStep(text=step)
