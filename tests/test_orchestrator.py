import json

import pytest

from physics_agent.llm_client import MockLLMClient
from physics_agent.orchestrator import ToolOrchestrator
from physics_agent.trace import Trace


def _make_trace(problem_text="A 2 kg block starts at rest at the top of a 5 m frictionless incline. Find its speed at the bottom.", domain_tags=None):
    trace = Trace.new(problem_text)
    trace.domain_tags = domain_tags or ["dynamics", "energy"]
    trace.subtasks = ["identify knowns", "apply conservation of energy", "solve for v"]
    trace.retrieved_knowledge = [
        {
            "id": "eng-003",
            "statement": "Conservation of mechanical energy: KE_i + U_i = KE_f + U_f",
            "conditions": "No non-conservative forces do work",
        }
    ]
    return trace


def test_orchestrator_run_populates_trace_with_defaults():
    llm = MockLLMClient()
    orchestrator = ToolOrchestrator(llm)
    trace = _make_trace()

    result = orchestrator.run(trace)

    assert result is trace  # mutates and returns the same trace
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].tool == "symbolic_math"
    assert trace.initial_solution is not None
    assert trace.orchestration_time_ms >= 0

    output = json.loads(trace.tool_calls[0].output)
    assert "error" not in output
    assert output["solutions_numeric"]


def test_orchestrator_filters_tools_not_offered_for_domain():
    # literature_search isn't hinted for "kinematics" -- if the mock model
    # tried to call it anyway, selection should drop it.
    canned = {
        "tool-selection component": (
            '{"tool_calls": [{"tool": "literature_search", "input": {"query": "x"}}, '
            '{"tool": "symbolic_math", "input": {"expression": "Eq(x,1)", "solve_for": "x"}}]}'
        )
    }
    llm = MockLLMClient(canned_responses=canned)
    orchestrator = ToolOrchestrator(llm)
    trace = _make_trace(domain_tags=["kinematics"])

    orchestrator.run(trace)

    tool_names = [tc.tool for tc in trace.tool_calls]
    assert "literature_search" not in tool_names
    assert "symbolic_math" in tool_names


def test_orchestrator_captures_tool_failure_without_raising():
    canned = {
        "tool-selection component": (
            '{"tool_calls": [{"tool": "symbolic_math", '
            '"input": {"expression": "not : valid ((( sympy", "solve_for": "x"}}]}'
        )
    }
    llm = MockLLMClient(canned_responses=canned)
    orchestrator = ToolOrchestrator(llm)
    trace = _make_trace()

    orchestrator.run(trace)  # should not raise

    assert len(trace.tool_calls) == 1
    output = json.loads(trace.tool_calls[0].output)
    assert "error" in output
    # synthesis should still run and produce something despite the tool failure
    assert trace.initial_solution is not None


def test_orchestrator_handles_zero_tool_calls():
    canned = {"tool-selection component": '{"tool_calls": []}'}
    llm = MockLLMClient(canned_responses=canned)
    orchestrator = ToolOrchestrator(llm)
    trace = _make_trace()

    orchestrator.run(trace)

    assert trace.tool_calls == []
    assert trace.initial_solution is not None  # synthesis still runs


def test_orchestrator_retries_on_bad_selection_json():
    # "valid JSON" must come first: it only appears in the orchestrator's
    # own retry-correction message, so it only matches (and returns a good
    # response) on the second attempt. "flaky selection" stays in the
    # conversation history throughout, so it must be checked second or it
    # would keep matching after the retry too.
    canned = {
        "valid JSON": '{"tool_calls": []}',
        "flaky selection": "not json at all",
    }
    llm = MockLLMClient(canned_responses=canned)
    orchestrator = ToolOrchestrator(llm, max_retries=1)
    trace = _make_trace(problem_text="flaky selection problem")

    orchestrator.run(trace)  # should recover via retry, not raise
    assert trace.tool_calls == []


def test_orchestrator_raises_after_exhausting_selection_retries():
    canned = {
        "unfixable selection": "still not json",
        "valid JSON": "still not json either",
    }
    llm = MockLLMClient(canned_responses=canned)
    orchestrator = ToolOrchestrator(llm, max_retries=1)
    trace = _make_trace(problem_text="unfixable selection problem")

    with pytest.raises(ValueError):
        orchestrator.run(trace)
