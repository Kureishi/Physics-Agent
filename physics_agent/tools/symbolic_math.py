"""
Symbolic Math Tool (Stage 2) -- the "Mathematical Reasoner" box in the
architecture diagram.

Wraps SymPy to solve algebraic equations, with optional numeric
substitution of knowns before solving.

Also handles a real-world pattern observed running this against an actual
local model (see the "note" field on the returned dict): a model
sometimes plugs every known value directly into `expression` and still
labels the target quantity as `solve_for`, even though it no longer
appears as a free symbol anywhere -- there's nothing left to *solve*, the
model just wants the arithmetic evaluated. Previously this raised
"SymPy found no solutions," causing the tool call to fail outright even
though the requested computation was perfectly well-defined; now it's
evaluated directly instead.
"""
from __future__ import annotations

from typing import Any, Dict

import sympy


class SymbolicMathTool:
    name = "symbolic_math"

    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expected input shape:
            {
              "expression": "Eq(0.5*m*v**2, m*g*h)",   # sympy-parseable equation
              "solve_for": "v",
              "substitutions": {"m": 2, "g": 9.8, "h": 5}   # optional
            }

        Returns:
            {
              "expression": "...", "solve_for": "...", "substitutions": {...},
              "solutions": [<str form>, ...],
              "solutions_numeric": [<float or None>, ...],
              "note": <str, only present when the request was evaluated
                       directly rather than solved -- see module docstring>
            }

        Raises ValueError on malformed input, unparseable expressions, or
        equations SymPy can't solve, so the orchestrator can capture the
        failure as a ToolCall with an error output instead of crashing.
        """
        expr_str = input.get("expression")
        solve_for = input.get("solve_for")
        substitutions = input.get("substitutions", {}) or {}

        if not expr_str or not solve_for:
            raise ValueError("symbolic_math requires 'expression' and 'solve_for'")

        try:
            expr = sympy.sympify(expr_str)
        except (sympy.SympifyError, SyntaxError, TypeError) as e:
            raise ValueError(f"Could not parse expression {expr_str!r}: {e}")

        try:
            symbol = sympy.symbols(solve_for)
        except Exception as e:
            raise ValueError(f"Could not parse solve_for variable {solve_for!r}: {e}")

        try:
            subs_symbols = {sympy.symbols(k): v for k, v in substitutions.items()}
        except Exception as e:
            raise ValueError(f"Could not parse substitutions {substitutions!r}: {e}")

        expr_substituted = expr.subs(subs_symbols) if subs_symbols else expr

        # If `solve_for` isn't actually a free symbol in the (substituted)
        # expression, and the expression is a bare arithmetic expression
        # (not an Eq(...)) with NO free symbols left at all, there is
        # nothing to solve -- evaluate it directly instead of erroring.
        # An Eq(...) with no matching free symbol -- including one that
        # auto-reduced to a plain True/False, e.g. Eq(1, 2) -> BooleanFalse,
        # which has no .evalf() -- is left to the normal solve() path below
        # (ambiguous enough there that erroring is still the right call).
        if (
            not isinstance(expr_substituted, sympy.Equality)
            and hasattr(expr_substituted, "evalf")
            and symbol not in expr_substituted.free_symbols
            and not expr_substituted.free_symbols
        ):
            value = expr_substituted.evalf()
            try:
                numeric_value = float(value)
            except TypeError:
                numeric_value = None
            return {
                "expression": expr_str,
                "solve_for": solve_for,
                "substitutions": substitutions,
                "solutions": [str(value)],
                "solutions_numeric": [numeric_value],
                "note": (
                    f"'{solve_for}' was not a free variable in the expression after "
                    "substitution; evaluated it directly instead of solving an equation."
                ),
            }

        try:
            solutions = sympy.solve(expr_substituted, symbol)
        except NotImplementedError as e:
            raise ValueError(f"SymPy could not solve for {solve_for}: {e}")

        if not solutions:
            raise ValueError(
                f"SymPy found no solutions for {solve_for} in {expr_str} "
                f"with substitutions {substitutions}"
            )

        solutions_numeric = []
        for s in solutions:
            try:
                solutions_numeric.append(float(s.evalf()) if s.is_number else None)
            except (TypeError, AttributeError):
                solutions_numeric.append(None)

        return {
            "expression": expr_str,
            "solve_for": solve_for,
            "substitutions": substitutions,
            "solutions": [str(s) for s in solutions],
            "solutions_numeric": solutions_numeric,
        }
