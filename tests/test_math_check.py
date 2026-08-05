import json

import pytest

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


# -- direct-evaluation false positive (found in a real run) ------------------
#
# SymbolicMathTool falls back to direct evaluation when solve_for was never
# actually a free symbol in the expression (e.g. "m * (v_f - v_i)" labeled
# "delta_p", which never appears in that expression). Re-substituting a
# symbol that isn't present is a silent SymPy no-op, so naively computing a
# "residual" against the unchanged expression produces a number that's
# essentially never zero -- failing this check on every direct evaluation
# regardless of correctness. Found in practice: 199 of 183 traces in one
# real run hit this fallback, and 190 (95.5%) were incorrectly flagged
# before this fix.


def test_math_check_accepts_correct_direct_evaluation_from_real_tool():
    # Exact reproduction of the real failing case: momentum change for a
    # baseball reversing direction. delta_p = m*(v_f - v_i) = 0.15*(-45-40)
    # = -12.75, which the tool correctly evaluates directly since delta_p
    # is just a label, not a variable in the expression.
    tool = SymbolicMathTool()
    output = tool.run(
        {
            "expression": "m * (v_f - v_i)",
            "solve_for": "delta_p",
            "substitutions": {"m": 0.15, "v_i": 40, "v_f": -45},
        }
    )
    assert output["solutions_numeric"][0] == pytest.approx(-12.75)

    tc = ToolCall(tool="symbolic_math", input="{}", output=json.dumps(output), latency_ms=1.0)
    check = MathCheck()
    result = check.run(_make_trace([tc]))

    assert result["passed"] is True  # previously always failed here
    assert "direct evaluation" in result["details"]


def test_math_check_accepts_direct_evaluation_even_with_manually_constructed_output():
    # Same shape as SymbolicMathTool's own fallback output, constructed
    # directly rather than via the tool, to pin the exact condition being
    # checked (presence of the "note" field isn't what's checked -- the
    # actual free-symbol condition is, so this must pass even without it).
    output = {
        "expression": "0.5 * m * v**2",
        "solve_for": "KE",  # never appears in the expression
        "substitutions": {"m": 2, "v": 3},
        "solutions": ["9.0"],
        "solutions_numeric": [9.0],
    }
    tc = ToolCall(tool="symbolic_math", input="{}", output=json.dumps(output), latency_ms=1.0)
    check = MathCheck()
    result = check.run(_make_trace([tc]))
    assert result["passed"] is True


def test_math_check_wrong_solution_in_an_equation_shape_still_fails():
    # NOT a direct-evaluation case -- delta_p appears on the RHS of this
    # Eq, so it goes through normal residual verification, same as any
    # other equation. Confirms the fix's skip condition (solve_for absent
    # as a free symbol, non-Eq expression) doesn't accidentally swallow a
    # genuinely wrong solution to a real equation just because the
    # variable names happen to resemble the direct-evaluation example above.
    bad_eq_output = {
        "expression": "Eq(m * (v_f - v_i), delta_p)",
        "solve_for": "delta_p",
        "substitutions": {"m": 0.15, "v_i": 40, "v_f": -45},
        "solutions": ["999"],  # wrong -- should be -12.75
    }
    tc = ToolCall(tool="symbolic_math", input="{}", output=json.dumps(bad_eq_output), latency_ms=1.0)
    check = MathCheck()
    result = check.run(_make_trace([tc]))
    assert result["passed"] is False  # genuine equation, genuinely wrong solution


def test_math_check_direct_evaluation_skip_does_not_apply_to_real_equations():
    # Sanity check the fix isn't overly broad: a genuine Eq(...) where
    # solve_for DOES appear must still go through full verification, not
    # get skipped just because it's possible to construct a similar shape.
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
    assert "verified by substitution" in result["details"]  # took the real path, not the skip
