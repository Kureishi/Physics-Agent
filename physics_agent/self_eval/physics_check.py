"""
Physics Check (Stage 3, extended in Stage 6) -- the "Physics Check" box in
the self-evaluation pipeline.

Three independent sub-checks, combined -- any of them failing fails the
overall check:
  1. Cross-tool agreement (deterministic, no LLM): if both a symbolic_math
     and a simulation tool call were made, do their numeric answers agree?
     This directly implements the "simulation and closed-form disagree"
     row of the self-correction mapping table from the design doc.
  2. Knowledge graph validity (Stage 6, deterministic, no LLM): for each
     retrieved fact, is it being used outside a known assumption it
     requires, given the problem's domain tags? This is the design doc's
     "check formula validity as a graph query instead of re-deriving it
     every time" -- narrow (see KnowledgeGraph.check_validity's docstring
     for exactly how narrow), but genuinely deterministic where it applies.
  3. Physics critique (LLM): reviews the solution for dimensional
     consistency and conservation-law violations that aren't captured by
     the two checks above.

If there aren't tool calls of both kinds to cross-check, sub-check 1 is
skipped. If no knowledge_graph was provided, sub-check 2 is skipped. The
LLM critique always runs.
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


def _knowledge_graph_validity_check(
    knowledge_graph, retrieved_knowledge: List[Dict[str, Any]], domain_tags: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Returns None if no knowledge_graph was provided (check doesn't apply).
    Otherwise checks every retrieved fact's requires_assumption edges
    against the problem's domain_tags and reports any violations.
    """
    if knowledge_graph is None:
        return None

    violations = []
    for fact in retrieved_knowledge:
        node_id = fact.get("id")
        if not node_id:
            continue
        result = knowledge_graph.check_validity(node_id, domain_tags)
        if not result["valid"]:
            violations.append((node_id, result["violated_assumptions"]))

    if not violations:
        return {
            "passed": True,
            "details": "Knowledge graph validity check: no assumption violations detected.",
        }

    detail_strs = [f"{node_id} violates assumption(s) {va}" for node_id, va in violations]
    return {
        "passed": False,
        "details": "Knowledge graph validity check failed: " + "; ".join(detail_strs),
    }


class PhysicsCheck:
    name = "physics"

    def __init__(self, llm_client, max_retries: int = 1, knowledge_graph=None):
        self.llm = llm_client
        self.max_retries = max_retries
        self.knowledge_graph = knowledge_graph

    def run(self, trace: Trace) -> Dict[str, Any]:
        agreement_result = _cross_tool_agreement_check(trace.tool_calls)
        kg_result = _knowledge_graph_validity_check(
            self.knowledge_graph, trace.retrieved_knowledge, trace.domain_tags
        )
        llm_result = self._llm_critique(trace)

        sub_results = [r for r in (agreement_result, kg_result, llm_result) if r is not None]
        passed = all(r["passed"] for r in sub_results)
        details = " | ".join(r["details"] for r in sub_results)
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
