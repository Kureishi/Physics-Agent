"""
Physics Check (Stage 3) -- the "Physics Check" box in the self-evaluation
pipeline.

Two independent sub-checks, combined:
  1. Cross-tool agreement (deterministic, no LLM): if both a symbolic_math
     and a simulation tool call were made, do their numeric answers agree?
     This directly implements the "simulation and closed-form disagree"
     row of the self-correction mapping table from the design doc.
  2. Physics critique (LLM): reviews the solution for dimensional
     consistency and conservation-law violations, using the conditions of
     validity attached to any retrieved formulas.

Either sub-check can fail the overall physics check; if there aren't tool
calls of both kinds to cross-check, only the LLM critique determines the
result.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..json_utils import extract_json
from ..trace import Trace, ToolCall

_SYSTEM_PROMPT = """You are the physics-check component of a physics problem-solving agent.

Given a problem, the physics facts retrieved from memory (with their
conditions of validity), and a proposed initial solution, check for:
- Dimensional inconsistency (units don't work out)
- Violated conservation laws (energy, momentum, charge, angular momentum)
  given the solution's own numbers
- Use of a formula outside its stated conditions of validity (e.g. a
  non-relativistic formula used at relativistic speeds)

Do NOT judge algebraic correctness -- only physics correctness. Algebra is
checked by a separate component.

Respond with ONLY valid JSON, no commentary, no markdown fences:
{"passed": true or false, "issues": ["issue 1", ...]}
"""

AGREEMENT_RELATIVE_TOLERANCE = 0.05  # 5%


def _cross_tool_agreement_check(tool_calls: List[ToolCall]) -> Optional[Dict[str, Any]]:
    """
    Returns None if there isn't at least one successful symbolic_math call
    and one successful simulation call to compare (the check doesn't
    apply). Otherwise returns {"passed": bool, "details": str}.
    """
    symbolic_values: List[float] = []
    simulation_values: Dict[str, float] = {}

    for tc in tool_calls:
        try:
            output = json.loads(tc.output)
        except json.JSONDecodeError:
            continue
        if "error" in output:
            continue

        if tc.tool == "symbolic_math":
            symbolic_values.extend(v for v in output.get("solutions_numeric", []) if v is not None)
        elif tc.tool == "simulation":
            simulation_values.update(output.get("final_state", {}))

    if not symbolic_values or not simulation_values:
        return None  # nothing to cross-check

    # We don't reliably know which simulation state variable corresponds
    # to the symbolic solve_for, so compare each symbolic value against
    # every simulation value and take the closest match.
    best = None
    for s_val in symbolic_values:
        for var_name, sim_val in simulation_values.items():
            if sim_val == 0:
                rel_diff = abs(s_val)
            else:
                rel_diff = abs(abs(s_val) - abs(sim_val)) / abs(sim_val)
            if best is None or rel_diff < best[0]:
                best = (rel_diff, s_val, var_name, sim_val)

    rel_diff, s_val, var_name, sim_val = best
    if rel_diff <= AGREEMENT_RELATIVE_TOLERANCE:
        return {
            "passed": True,
            "details": (
                f"symbolic_math result {s_val} agrees with simulation's "
                f"{var_name}={sim_val} (relative difference {rel_diff:.1%})"
            ),
        }
    return {
        "passed": False,
        "details": (
            f"symbolic_math result {s_val} disagrees with simulation's "
            f"{var_name}={sim_val} (relative difference {rel_diff:.1%}, "
            f"exceeds {AGREEMENT_RELATIVE_TOLERANCE:.0%} tolerance)"
        ),
    }


class PhysicsCheck:
    name = "physics"

    def __init__(self, llm_client, max_retries: int = 1):
        self.llm = llm_client
        self.max_retries = max_retries

    def run(self, trace: Trace) -> Dict[str, Any]:
        agreement_result = _cross_tool_agreement_check(trace.tool_calls)
        llm_result = self._llm_critique(trace)

        if agreement_result is None:
            return llm_result

        passed = agreement_result["passed"] and llm_result["passed"]
        details = agreement_result["details"] + " | " + llm_result["details"]
        return {"passed": passed, "details": details}

    def _llm_critique(self, trace: Trace) -> Dict[str, Any]:
        user_content = json.dumps(
            {
                "problem": trace.problem_text,
                "retrieved_knowledge": [
                    {"statement": k["statement"], "conditions": k["conditions"]}
                    for k in trace.retrieved_knowledge
                ],
                "initial_solution": trace.initial_solution,
            }
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw = ""
        last_err: Exception = ValueError("physics check never ran")
        for _ in range(self.max_retries + 1):
            raw = self.llm.chat(messages)
            try:
                parsed = extract_json(raw)
                passed = bool(parsed.get("passed", False))
                issues = [i for i in parsed.get("issues", []) if isinstance(i, str)]
                details = "; ".join(issues) if issues else "No physics issues found."
                return {"passed": passed, "details": details}
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That was not valid JSON in the required shape. "
                            "Reply again with ONLY the JSON object."
                        ),
                    }
                )

        return {
            "passed": False,
            "details": f"Physics check failed to parse model output: {last_err}",
        }
