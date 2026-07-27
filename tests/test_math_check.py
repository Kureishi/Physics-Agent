import json

from physics_agent.self_eval.math_check import MathCheck
from physics_agent.tools.symbolic_math import SymbolicMathTool
from physics_agent.trace import Trace, ToolCall


def _make_trace(tool_calls=None):
    trace = Trace.new("test problem")
    trace.tool_calls = tool_calls or []
    return trace


def test_math_check_passes_with_no_symbolic_calls():
    check = MathCheck()
    result = check.run(_make_trace())
    assert result["passed"] is True
    assert "No symbolic math" in result["details"]


def test_math_check_verifies_correct_solution_from_real_tool():
    # Use the real SymbolicMathTool to produce output, then verify the math
    # check accepts its own (correct) solution.
    tool = SymbolicMathTool()
    output = tool.run(
        {
            "expression": "Eq(m*g*h, 0.5*m*v**2)",
            "solve_for": "v",
            "substitutions": {"m": 2, "g": 9.8, "h": 5},
        }
    )
    tc = ToolCall(tool="symbolic_math", input="{}", output=json.dumps(output), latency_ms=1.0)

    check = MathCheck()
    result = check.run(_make_trace([tc]))
    assert result["passed"] is True
    assert "verified by substitution" in result["details"]


def test_math_check_fails_on_wrong_solution():
    # Manually construct a tool output claiming a wrong solution (v=100
    # does not satisfy the energy conservation equation for these values).
    bad_output = {
        "expression": "Eq(m*g*h, 0.5*m*v**2)",
        "solve_for": "v",
        "substitutions": {"m": 2, "g": 9.8, "h": 5},
        "solutions": ["100"],
    }
    tc = ToolCall(tool="symbolic_math", input="{}", output=json.dumps(bad_output), latency_ms=1.0)

    check = MathCheck()
    result = check.run(_make_trace([tc]))
    assert result["passed"] is False
    assert "does not satisfy" in result["details"]


def test_math_check_ignores_errored_tool_calls():
    errored = ToolCall(tool="symbolic_math", input="{}", output=json.dumps({"error": "parse failed"}), latency_ms=1.0)
    check = MathCheck()
    result = check.run(_make_trace([errored]))
    assert result["passed"] is True
    assert "No successful" in result["details"]


def test_math_check_handles_multiple_calls_mixed_pass_fail():
    tool = SymbolicMathTool()
    good_output = tool.run(
        {"expression": "Eq(x, 5)", "solve_for": "x", "substitutions": {}}
    )
    good_tc = ToolCall(tool="symbolic_math", input="{}", output=json.dumps(good_output), latency_ms=1.0)

    bad_output = {
        "expression": "Eq(x, 5)",
        "solve_for": "x",
        "substitutions": {},
        "solutions": ["999"],
    }
    bad_tc = ToolCall(tool="symbolic_math", input="{}", output=json.dumps(bad_output), latency_ms=1.0)

    check = MathCheck()
    result = check.run(_make_trace([good_tc, bad_tc]))
    assert result["passed"] is False  # one bad call fails the whole check
