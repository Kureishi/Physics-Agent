from physics_agent.llm_client import MockLLMClient
from physics_agent.self_eval.logic_check import LogicCheck
from physics_agent.trace import Trace


def _make_trace():
    trace = Trace.new("A ball is dropped from 10 m. Find its speed at impact.")
    trace.subtasks = ["identify knowns", "apply kinematics", "solve for v"]
    trace.initial_solution = "Using v^2 = 2*g*h, v = sqrt(2*9.8*10) = 14.0 m/s."
    return trace


def test_logic_check_default_passes():
    llm = MockLLMClient()
    check = LogicCheck(llm)
    result = check.run(_make_trace())
    assert result["passed"] is True


def test_logic_check_reports_issues_on_failure():
    canned = {
        "logic-check component": (
            '{"passed": false, "issues": ["final answer contradicts stated approach"]}'
        )
    }
    llm = MockLLMClient(canned_responses=canned)
    check = LogicCheck(llm)
    result = check.run(_make_trace())
    assert result["passed"] is False
    assert "contradicts" in result["details"]


def test_logic_check_recovers_via_retry():
    canned = {
        "not valid JSON": '{"passed": true, "issues": []}',
        "flaky logic problem": "not json",
    }
    llm = MockLLMClient(canned_responses=canned)
    check = LogicCheck(llm, max_retries=1)
    trace = _make_trace()
    trace.problem_text = "flaky logic problem"
    result = check.run(trace)
    assert result["passed"] is True
    assert len(llm.calls) == 2


def test_logic_check_fails_gracefully_after_exhausted_retries():
    canned = {
        "unfixable logic": "still not json",
        "valid JSON": "still not json either",
    }
    llm = MockLLMClient(canned_responses=canned)
    check = LogicCheck(llm, max_retries=1)
    trace = _make_trace()
    trace.problem_text = "unfixable logic problem"
    result = check.run(trace)  # should not raise
    assert result["passed"] is False
    assert "failed to parse" in result["details"]
