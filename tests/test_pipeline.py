import json

from conftest import ScriptedLLM

from reversegpt import mock_server

ISSUE = "don:core:dvrv-us-1:devo/0:issue/1"


def args_of(call: dict) -> dict:
    return {a["argument_name"]: a["argument_value"] for a in call["arguments"]}


def test_two_step_plan_links_outputs(make_agent):
    llm = ScriptedLLM(
        plan=[
            f"Plan step 1: Get the work items similar to {ISSUE}.",
            "Summarize the similar work items ($$PREV[0]).",
            "Thought: The similar issues are summarized.\nFinal Answer: Here is the summary.",
        ],
        select={
            f"Get the work items similar to {ISSUE}.": "get_similar_work_items",
            "Summarize the similar work items ($$PREV[0]).": "summarize_objects",
        },
        arguments={
            "get_similar_work_items": json.dumps({"work_id": ISSUE}),
            "summarize_objects": '```json\n{"objects": "$$PREV[0]"}\n```',
        },
    )
    result = make_agent(llm).run(f"Summarize issues similar to {ISSUE}")

    assert result.status == "done"
    assert result.answer == "Here is the summary."
    assert result.tool_calls() == [
        {
            "tool_name": "get_similar_work_items",
            "arguments": [{"argument_name": "work_id", "argument_value": ISSUE}],
        },
        {
            "tool_name": "summarize_objects",
            "arguments": [{"argument_name": "objects", "argument_value": "$$PREV[0]"}],
        },
    ]
    # The executor resolved the reference before calling the summarizer.
    similar = mock_server.get_similar_work_items(ISSUE)["similar_work_items"]
    assert result.calls[1].outputs["summary"].startswith(f"{len(similar)} objects")


def test_reverse_chain_resolves_missing_arguments_backwards(make_agent):
    step = "Add my prioritized P0 issues to the current sprint."
    llm = ScriptedLLM(
        plan=[step, "Final Answer: Done."],
        select={step: "add_work_items_to_sprint"},
        arguments={
            "add_work_items_to_sprint": '{"work_ids": null, "sprint_id": null}',
            "prioritize_objects": '{"objects": null}',
            "work_list": '{"issue.priority": ["p0"], "type": ["issue"], "limit": null}',
        },
        providers={
            ("add_work_items_to_sprint", "work_ids"): "prioritize_objects",
            ("prioritize_objects", "objects"): "work_list",
            # sprint_id is found by the exact output-name shortcut, no LLM call
        },
    )
    result = make_agent(llm).run("Prioritize my P0 issues and add them to the current sprint")

    calls = result.tool_calls()
    assert [c["tool_name"] for c in calls] == [
        "work_list",
        "prioritize_objects",
        "get_sprint_id",
        "add_work_items_to_sprint",
    ]
    assert args_of(calls[0]) == {"issue.priority": ["p0"], "type": ["issue"]}
    assert args_of(calls[1]) == {"objects": "$$PREV[0]"}
    assert args_of(calls[2]) == {}
    assert args_of(calls[3]) == {"work_ids": "$$PREV[1]", "sprint_id": "$$PREV[2]"}
    assert not any('needs the argument "sprint_id"' in p for p in llm.prompts)


def test_scalar_reference_is_wrapped_for_array_argument(make_agent):
    llm = ScriptedLLM(
        plan=[
            "Get the ID of the current user.",
            "Get P0 issues owned by $$PREV[0].",
            "Final Answer: ok",
        ],
        select={
            "Get the ID of the current user.": "who_am_i",
            "Get P0 issues owned by $$PREV[0].": "work_list",
        },
        arguments={"work_list": '{"owned_by": "$$PREV[0]", "issue.priority": "p0"}'},
    )
    result = make_agent(llm).run("What are my P0 issues?")

    calls = result.tool_calls()
    assert args_of(calls[1]) == {"owned_by": ["$$PREV[0]"], "issue.priority": ["p0"]}


def test_pasted_values_are_turned_back_into_references(make_agent):
    similar = mock_server.get_similar_work_items(ISSUE)["similar_work_items"]
    llm = ScriptedLLM(
        plan=[f"Get items similar to {ISSUE}", "Prioritize them", "Final Answer: ok"],
        select={
            f"Get items similar to {ISSUE}": "get_similar_work_items",
            "Prioritize them": "prioritize_objects",
        },
        arguments={
            "get_similar_work_items": json.dumps({"work_id": ISSUE}),
            "prioritize_objects": json.dumps({"objects": similar}),
        },
    )
    result = make_agent(llm).run("Prioritize issues similar to issue 1")
    assert args_of(result.tool_calls()[1]) == {"objects": "$$PREV[0]"}


def test_existing_output_is_reused_instead_of_calling_again(make_agent):
    llm = ScriptedLLM(
        plan=["Get the current sprint.", "Add the tasks to the sprint.", "Final Answer: ok"],
        select={
            "Get the current sprint.": "get_sprint_id",
            "Add the tasks to the sprint.": "add_work_items_to_sprint",
        },
        arguments={"add_work_items_to_sprint": '{"work_ids": ["TASK-1"], "sprint_id": null}'},
    )
    result = make_agent(llm).run("Add TASK-1 to the current sprint")
    calls = result.tool_calls()
    assert [c["tool_name"] for c in calls] == ["get_sprint_id", "add_work_items_to_sprint"]
    assert args_of(calls[1])["sprint_id"] == "$$PREV[0]"


def test_unrelated_query_returns_empty_list(make_agent):
    result = make_agent(ScriptedLLM(plan=["[]"])).run("What's the weather in Paris?")
    assert result.status == "unrelated"
    assert result.tool_calls() == []


def test_missing_provider_reports_failure_to_planner(make_agent):
    step = "Summarize the work items."
    llm = ScriptedLLM(
        plan=[step, "Final Answer: gave up"],
        select={step: "summarize_objects"},
        arguments={"summarize_objects": '{"objects": null}'},
        providers={},  # nothing produces "objects"
    )
    result = make_agent(llm).run("Summarize")
    assert result.tool_calls() == []
    assert "could not run" in result.steps[0][1]


def test_stops_at_max_plan_steps(make_agent):
    llm = ScriptedLLM(plan=["Who am I?"] * 10, select={"Who am I?": "who_am_i"})
    result = make_agent(llm, max_plan_steps=3).run("loop forever")
    assert result.status == "max_steps"
    assert len(result.steps) == 3


def test_continue_retries_step_with_another_tool(make_agent):
    step = "Get my issues."
    llm = ScriptedLLM(plan=[step, "Continue", "Final Answer: ok"], select={step: "work_list"})
    agent = make_agent(llm)
    result = agent.run("my issues")
    # The retry excluded work_list, so the selector saw no matching name and nothing ran twice.
    assert [c["tool_name"] for c in result.tool_calls()] == ["work_list"]
    retry_prompt = [p for p in llm.prompts if p.startswith("Pick the API")][1]
    assert "API: work_list" not in retry_prompt
