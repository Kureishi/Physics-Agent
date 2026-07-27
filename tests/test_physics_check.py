import json

from physics_agent.llm_client import MockLLMClient
from physics_agent.self_eval.physics_check import PhysicsCheck
from physics_agent.trace import Trace, ToolCall


def _make_trace(tool_calls=None):
    trace = Trace.new("A 2 kg block starts at rest atop a 5 m frictionless incline.")
    trace.retrieved_knowledge = [
        {
            "statement": "KE = 0.5*m*v^2",
            "conditions": "Non-relativistic",
        }
    ]
    trace.initial_solution = "v = sqrt(2*g*h) = 9.9 m/s"
    trace.tool_calls = tool_calls or []
    return trace


def _symbolic_call(value):
    output = {
        "expression": "Eq(m*g*h, 0.5*m*v**2)",
        "solve_for": "v",
        "solutions_numeric": [value],
    }
    return ToolCall(tool="symbolic_math", input="{}", output=json.dumps(output), latency_ms=1.0)


def _simulation_call(final_state):
    output = {"final_state": final_state}
    return ToolCall(tool="simulation", input="{}", output=json.dumps(output), latency_ms=1.0)


def test_physics_check_passes_when_no_tools_to_cross_check():
    llm = MockLLMClient()  # default physics response passes
    check = PhysicsCheck(llm)
    result = check.run(_make_trace())
    assert result["passed"] is True


def test_physics_check_passes_when_symbolic_and_simulation_agree():
    llm = MockLLMClient()
    check = PhysicsCheck(llm)
    tool_calls = [_symbolic_call(9.9), _simulation_call({"v": 9.85, "x": 5.0})]
    result = check.run(_make_trace(tool_calls))
    assert result["passed"] is True
    assert "agrees" in result["details"]


def test_physics_check_fails_when_symbolic_and_simulation_disagree():
    llm = MockLLMClient()
    check = PhysicsCheck(llm)
    tool_calls = [_symbolic_call(9.9), _simulation_call({"v": 3.0})]
    result = check.run(_make_trace(tool_calls))
    assert result["passed"] is False
    assert "disagrees" in result["details"]


def test_physics_check_fails_when_llm_critique_fails_even_if_tools_agree():
    canned = {"physics-check component": '{"passed": false, "issues": ["formula used outside validity range"]}'}
    llm = MockLLMClient(canned_responses=canned)
    check = PhysicsCheck(llm)
    tool_calls = [_symbolic_call(9.9), _simulation_call({"v": 9.9})]
    result = check.run(_make_trace(tool_calls))
    assert result["passed"] is False
    assert "validity range" in result["details"]


def test_physics_check_ignores_errored_tool_calls_for_cross_check():
    llm = MockLLMClient()
    check = PhysicsCheck(llm)
    errored = ToolCall(tool="symbolic_math", input="{}", output=json.dumps({"error": "bad input"}), latency_ms=1.0)
    result = check.run(_make_trace([errored]))
    # no valid symbolic result -> cross-check doesn't apply -> falls back to LLM critique (passes by default)
    assert result["passed"] is True
