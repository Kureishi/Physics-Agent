"""
Math Check (Stage 3) -- the "Math Check" box in the self-evaluation
pipeline.

Purely deterministic, no LLM involved: re-substitutes each symbolic_math
tool call's reported solution back into its original equation and verifies
the equation actually holds. This catches algebra errors independent of
whether the underlying physics setup was correct (that's PhysicsCheck's job).

Does NOT attempt re-substitution verification for a direct-evaluation
result -- SymbolicMathTool's fallback for when `solve_for` was never
actually a free symbol in the (substituted) expression, e.g. a model
computing "m * (v_f - v_i)" and labeling the result "delta_p" even though
delta_p never appears in that expression at all. That fallback's "solution"
IS the expression's own value by definition; there's no equation
constraint it's meant to satisfy. Re-substituting a symbol that was never
present is a silent SymPy no-op, so naively residual-checking it computes
the expression's own unchanged value as the "residual" -- a number that's
essentially never zero, which would fail this check on every direct
evaluation regardless of whether the computed value is correct. This was
found to be a real, severe false positive in practice: 199 of 183 traces
in one real run triggered the tool's evaluate-fallback, and 190 of those
(95.5%) were incorrectly flagged as a math failure by this check before
the fix below, corrupting the self-correction loop's revision behavior,
semantic-memory confidence updates, and every downstream Stage 7 statistic
that reads from them.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import sympy

from ..trace import Trace

NUMERIC_TOLERANCE = 1e-6


class MathCheck:
    name = "math"

    def run(self, trace: Trace) -> Dict[str, Any]:
        symbolic_calls = [tc for tc in trace.tool_calls if tc.tool == "symbolic_math"]
        if not symbolic_calls:
            return {"passed": True, "details": "No symbolic math tool calls to verify."}

        all_details: List[str] = []
        all_passed = True
        any_verifiable = False

        for tc in symbolic_calls:
            try:
                output = json.loads(tc.output)
            except json.JSONDecodeError:
                all_details.append(f"Could not parse tool output for verification: {tc.output!r}")
                all_passed = False
                continue

            if "error" in output:
                # Already visible as a tool failure elsewhere in the
                # trace; not something the math check re-litigates.
                continue

            any_verifiable = True
            verified = self._verify_solutions(output)
            all_details.append(verified["details"])
            if not verified["passed"]:
                all_passed = False

        if not any_verifiable:
            return {"passed": True, "details": "No successful symbolic math results to verify."}

        details = " | ".join(all_details)
        return {"passed": all_passed, "details": details}

    def _verify_solutions(self, output: Dict[str, Any]) -> Dict[str, Any]:
        expr_str = output.get("expression")
        solve_for = output.get("solve_for")
        substitutions = output.get("substitutions", {}) or {}
        solutions = output.get("solutions", [])

        if not expr_str or not solve_for or not solutions:
            return {"passed": False, "details": "Incomplete symbolic_math output; cannot verify."}

        try:
            expr = sympy.sympify(expr_str)
            symbol = sympy.symbols(solve_for)
            subs_symbols = {sympy.symbols(k): v for k, v in substitutions.items()}
        except (sympy.SympifyError, SyntaxError, TypeError) as e:
            return {"passed": False, "details": f"Could not re-parse expression for verification: {e}"}

        # Same condition SymbolicMathTool itself checks before falling back
        # to direct evaluation (see its docstring): a bare expression (not
        # an Eq) where solve_for was never a free symbol to begin with.
        # There's no equation constraint to re-verify against here -- the
        # reported "solution" IS just this expression's value.
        expr_substituted_for_check = expr.subs(subs_symbols) if subs_symbols else expr
        if (
            not isinstance(expr, sympy.Equality)
            and symbol not in expr.free_symbols
            and not expr_substituted_for_check.free_symbols
        ):
            return {
                "passed": True,
                "details": (
                    f"'{solve_for}' was a direct evaluation of {expr_str}, not a solved "
                    "equation -- re-substitution verification doesn't apply."
                ),
            }

        # For Eq(a, b), verify a - b == 0 after substitution. A bare
        # expression (no Eq) is checked against zero directly.
        if isinstance(expr, sympy.Equality):
            residual_expr = expr.lhs - expr.rhs
        else:
            residual_expr = expr

        failures = []
        for sol_str in solutions:
            try:
                sol_expr = sympy.sympify(sol_str)
            except (sympy.SympifyError, SyntaxError, TypeError):
                failures.append(f"could not re-parse solution {sol_str!r}")
                continue

            full_subs = dict(subs_symbols)
            full_subs[symbol] = sol_expr
            residual = residual_expr.subs(full_subs)

            try:
                residual_value = complex(residual.evalf())
            except (TypeError, AttributeError):
                # Still has unresolved free symbols -- can't numerically
                # verify; treat as unverifiable rather than pass or fail.
                failures.append(
                    f"solution {sol_str} has unresolved free symbols; skipped verification"
                )
                continue

            if abs(residual_value) > NUMERIC_TOLERANCE:
                failures.append(
                    f"solution {sol_str} does not satisfy {expr_str} (residual={residual_value})"
                )

        if failures:
            return {"passed": False, "details": "; ".join(failures)}
        return {
            "passed": True,
            "details": f"All {len(solutions)} solution(s) for {expr_str} verified by substitution.",
        }
