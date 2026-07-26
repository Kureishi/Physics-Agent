"""
Simulation Tool (Stage 2) -- the "Simulation Engine" box in the
architecture diagram.

Numerically integrates a system of first-order ODEs defined symbolically.
Used both to solve problems without a clean closed form, and to
cross-check closed-form symbolic answers (self-eval, Stage 3, can compare
symbolic_math's answer against this tool's independent numerical answer).
"""
from __future__ import annotations

from typing import Any, Dict

import sympy
from scipy.integrate import solve_ivp


class SimulationTool:
    name = "simulation"

    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expected input shape:
            {
              "state_vars": ["x", "v"],
              "derivatives": ["v", "-g"],        # dx/dt, dv/dt as sympy-parseable strings,
                                                   # each may reference state_vars and params
              "params": {"g": 9.8},
              "initial_conditions": {"x": 0, "v": 10},
              "t_span": [0, 2],
              "t_eval": [2]                       # optional; times to report
            }

        Returns final state and full trajectory.
        Raises ValueError on malformed input or a failed integration.
        """
        state_vars = input.get("state_vars")
        derivatives = input.get("derivatives")
        params = input.get("params", {}) or {}
        initial_conditions = input.get("initial_conditions")
        t_span = input.get("t_span")

        if not state_vars or not derivatives or not initial_conditions or not t_span:
            raise ValueError(
                "simulation requires 'state_vars', 'derivatives', "
                "'initial_conditions', and 't_span'"
            )
        if len(state_vars) != len(derivatives):
            raise ValueError("state_vars and derivatives must be the same length")

        state_symbols = sympy.symbols(state_vars)
        if not isinstance(state_symbols, (list, tuple)):
            state_symbols = (state_symbols,)
        param_symbols = {name: sympy.symbols(name) for name in params}

        local_dict = {s.name: s for s in state_symbols}
        local_dict.update(param_symbols)

        try:
            derivative_exprs = [sympy.sympify(d, locals=local_dict) for d in derivatives]
        except (sympy.SympifyError, SyntaxError, TypeError) as e:
            raise ValueError(f"Could not parse derivative expressions {derivatives!r}: {e}")

        all_symbols = list(state_symbols) + list(param_symbols.values())
        try:
            lambdified = [sympy.lambdify(all_symbols, expr, "numpy") for expr in derivative_exprs]
        except Exception as e:
            raise ValueError(f"Could not lambdify derivative expressions: {e}")

        param_values = [params[name] for name in params]

        def rhs(t, y):
            return [f(*y, *param_values) for f in lambdified]

        try:
            y0 = [initial_conditions[str(s)] for s in state_symbols]
        except KeyError as e:
            raise ValueError(f"Missing initial condition for state variable {e}")

        t_eval = input.get("t_eval")
        try:
            result = solve_ivp(
                rhs, t_span, y0, t_eval=t_eval, dense_output=False, rtol=1e-8, atol=1e-10
            )
        except Exception as e:
            raise ValueError(f"Simulation raised an exception during integration: {e}")

        if not result.success:
            raise ValueError(f"Simulation failed to integrate: {result.message}")

        final_state = {str(s): float(result.y[i][-1]) for i, s in enumerate(state_symbols)}

        return {
            "state_vars": state_vars,
            "final_time": float(result.t[-1]),
            "final_state": final_state,
            "t_values": result.t.tolist(),
            "trajectory": {str(s): result.y[i].tolist() for i, s in enumerate(state_symbols)},
        }
