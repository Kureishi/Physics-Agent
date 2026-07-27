"""
Logic Check (Stage 3) -- the "Logic Check" box in the self-evaluation
pipeline of the architecture diagram.

Asks the LLM to review whether the initial solution's reasoning is
internally consistent and actually follows from the stated subtasks --
independent of whether the physics or algebra is correct (those are
PhysicsCheck's and MathCheck's jobs respectively). Catches things like the
final answer contradicting an earlier step, an unjustified logical leap, or
a subtask that was listed but never actually addressed.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from ..json_utils import extract_json
from ..trace import Trace

_SYSTEM_PROMPT = """You are the logic-check component of a physics problem-solving agent.

Given a problem, its planned subtasks, and a proposed initial solution,
determine whether the solution's REASONING is internally consistent:
- Does it actually address each subtask, or skip steps unjustifiably?
- Does the final answer follow from the stated reasoning, without contradiction?
- Are there any logical leaps not supported by what came before?

Do NOT judge whether the underlying physics or algebra is correct -- only
whether the reasoning is internally consistent and complete. Physics and
math correctness are checked by separate components.

Respond with ONLY valid JSON, no commentary, no markdown fences:
{"passed": true or false, "issues": ["issue 1", "issue 2", ...]}
If there are no issues, use an empty list for "issues".
"""


class LogicCheck:
    name = "logic"

    def __init__(self, llm_client, max_retries: int = 1):
        self.llm = llm_client
        self.max_retries = max_retries

    def run(self, trace: Trace) -> Dict[str, Any]:
        user_content = json.dumps(
            {
                "problem": trace.problem_text,
                "subtasks": trace.subtasks,
                "initial_solution": trace.initial_solution,
            }
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw = ""
        last_err: Exception = ValueError("logic check never ran")
        for _ in range(self.max_retries + 1):
            raw = self.llm.chat(messages)
            try:
                parsed = extract_json(raw)
                passed = bool(parsed.get("passed", False))
                issues = [i for i in parsed.get("issues", []) if isinstance(i, str)]
                details = "; ".join(issues) if issues else "No logical issues found."
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

        # A check that can't even parse its own model's response is itself
        # a failure signal, surfaced as a failed check rather than crashing
        # the whole self-evaluation pipeline.
        return {
            "passed": False,
            "details": f"Logic check failed to parse model output: {last_err}",
        }
