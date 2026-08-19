import json

import pytest

from physics_agent.llm_client import MockLLMClient
from physics_agent.orchestrator import ToolOrchestrator
from physics_agent.tools.literature import LiteratureSearchTool
from physics_agent.tools.registry import ToolRegistry
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


class _CountingLLM:
    """Returns configured responses in order (by call count); extra calls
    beyond the list repeat the last response. Tracks the temperature each
    call was made with, so retry behavior is directly observable."""

    def __init__(self, responses):
        self.responses = responses
        self.temperatures = []

    def chat(self, messages, temperature=None):
        self.temperatures.append(temperature)
        idx = min(len(self.temperatures) - 1, len(self.responses) - 1)
        return self.responses[idx]


def test_synthesis_retries_on_empty_response_and_recovers():
    llm = _CountingLLM(["", "A real synthesized solution. v = 9.9 m/s."])
    orchestrator = ToolOrchestrator(llm, max_retries=1)
    trace = _make_trace()

    result = orchestrator._synthesize_solution(trace)

    assert result == "A real synthesized solution. v = 9.9 m/s."
    assert len(llm.temperatures) == 2
    assert llm.temperatures[0] is None  # first attempt: caller's own default temperature
    assert llm.temperatures[1] == pytest.approx(0.1)  # retry: lower temperature


def test_synthesis_raises_after_exhausting_retries_on_persistent_empty():
    llm = _CountingLLM([""])  # always empty
    orchestrator = ToolOrchestrator(llm, max_retries=1)
    trace = _make_trace()

    with pytest.raises(RuntimeError, match="empty response"):
        orchestrator._synthesize_solution(trace)

    assert len(llm.temperatures) == 2  # first attempt + 1 retry, then give up


def test_synthesis_no_retry_needed_when_first_response_is_nonempty():
    llm = _CountingLLM(["A solution on the first try."])
    orchestrator = ToolOrchestrator(llm, max_retries=1)
    trace = _make_trace()

    result = orchestrator._synthesize_solution(trace)
    assert result == "A solution on the first try."
    assert len(llm.temperatures) == 1


def test_synthesis_retries_on_whitespace_only_response():
    llm = _CountingLLM(["   \n  ", "Real content now."])
    orchestrator = ToolOrchestrator(llm, max_retries=1)
    trace = _make_trace()

    result = orchestrator._synthesize_solution(trace)
    assert result == "Real content now."


def test_synthesis_retries_on_raised_exception_then_recovers():
    class _RaisesOnceLLM:
        def __init__(self):
            self.n_calls = 0

        def chat(self, messages, temperature=None):
            self.n_calls += 1
            if self.n_calls == 1:
                raise ConnectionError("simulated network failure")
            return "Recovered after an exception."

    llm = _RaisesOnceLLM()
    orchestrator = ToolOrchestrator(llm, max_retries=1)
    trace = _make_trace()

    result = orchestrator._synthesize_solution(trace)
    assert result == "Recovered after an exception."


def test_synthesis_uses_configured_retry_temperature():
    llm = _CountingLLM(["", "ok"])
    orchestrator = ToolOrchestrator(llm, max_retries=1, synthesis_retry_temperature=0.05)
    trace = _make_trace()

    orchestrator._synthesize_solution(trace)
    assert llm.temperatures[1] == pytest.approx(0.05)


def test_synthesis_max_retries_zero_means_single_attempt():
    llm = _CountingLLM([""])
    orchestrator = ToolOrchestrator(llm, max_retries=0)
    trace = _make_trace()

    with pytest.raises(RuntimeError):
        orchestrator._synthesize_solution(trace)
    assert len(llm.temperatures) == 1


def test_mock_llm_client_never_triggers_the_empty_response_path():
    # Sanity check: MockLLMClient's DEFAULT_SYNTHESIS_RESPONSE is always
    # non-empty, so --dry-run / offline tests never exercise the retry
    # loop -- this behavior change should be invisible to anything running
    # against the mock.
    llm = MockLLMClient()
    orchestrator = ToolOrchestrator(llm)
    trace = _make_trace()

    result = orchestrator._synthesize_solution(trace)
    assert result == MockLLMClient.DEFAULT_SYNTHESIS_RESPONSE


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


def test_orchestrator_run_threads_feedback_into_prompts():
    llm = MockLLMClient()
    orchestrator = ToolOrchestrator(llm)
    trace = _make_trace()

    orchestrator.run(trace, feedback="Previous attempt had a sign error.")

    # feedback should appear in at least one of the calls made (tool
    # selection and/or synthesis)
    all_text = "\n".join(
        m["content"] for call in llm.calls for m in call
    )
    assert "Previous attempt had a sign error." in all_text


def test_orchestrator_resynthesize_does_not_touch_tool_calls():
    llm = MockLLMClient()
    orchestrator = ToolOrchestrator(llm)
    trace = _make_trace()
    orchestrator.run(trace)  # establish an initial state with tool calls

    original_tool_calls = list(trace.tool_calls)
    new_solution = orchestrator.resynthesize(trace, feedback="Fix the reasoning gap.")

    assert trace.tool_calls == original_tool_calls  # untouched
    assert trace.initial_solution == new_solution


def test_orchestrator_escalate_with_literature_search_appends_tool_call():
    llm = MockLLMClient()
    fake_arxiv_response = (
        '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    )
    registry = ToolRegistry()
    registry._tools["literature_search"] = LiteratureSearchTool(
        fetch_fn=lambda url: fake_arxiv_response
    )
    orchestrator = ToolOrchestrator(llm, registry=registry)
    trace = _make_trace()
    orchestrator.run(trace)  # establish an initial state

    n_calls_before = len(trace.tool_calls)
    orchestrator.escalate_with_literature_search(trace, feedback="Low confidence, no specific fault.")

    assert len(trace.tool_calls) == n_calls_before + 1
    assert trace.tool_calls[-1].tool == "literature_search"


def test_orchestrator_tool_policy_filters_offered_tools():
    llm = MockLLMClient()

    class StubToolPolicy:
        def filter_tools(self, domain_tags, available_tools):
            return [t for t in available_tools if t != "simulation"]

    orchestrator = ToolOrchestrator(llm, tool_policy=StubToolPolicy())
    trace = _make_trace(domain_tags=["dynamics"])  # normally offers both symbolic_math and simulation

    orchestrator.run(trace)

    # The "Available tools for this problem: [...]" line is what actually
    # constrains selection; the prompt separately documents all three
    # tools' input formats unconditionally, so we check that specific line
    # rather than the whole prompt text.
    selection_call = llm.calls[0]
    system_prompt = selection_call[0]["content"]
    available_line = next(line for line in system_prompt.splitlines() if line.startswith("Available tools"))
    assert "simulation" not in available_line
    assert "symbolic_math" in available_line


def test_orchestrator_without_tool_policy_offers_full_default_set():
    llm = MockLLMClient()
    orchestrator = ToolOrchestrator(llm)  # no tool_policy -- pre-Stage-7 behavior
    trace = _make_trace(domain_tags=["dynamics"])

    orchestrator.run(trace)

    selection_call = llm.calls[0]
    system_prompt = selection_call[0]["content"]
    available_line = next(line for line in system_prompt.splitlines() if line.startswith("Available tools"))
    assert "simulation" in available_line
