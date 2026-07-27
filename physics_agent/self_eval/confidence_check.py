"""
Confidence Check (Stage 3) -- the "Confidence Check" box in the
self-evaluation pipeline.

Asks the LLM for a confidence estimate given the problem, the initial
solution, and the outcome of the Logic/Physics/Math checks that already
ran. Below `threshold`, the check itself is marked failed -- not because
anything was definitively wrong, but as a signal to escalate (Stage 5:
deeper verification, or human review) rather than silently shipping a
low-confidence answer.

This is also where trace.final_confidence gets its first value. Stage 5
self-correction may revise it after a correction pass; Stage 7
meta-learning is what calibrates whether this number is trustworthy over
time (tracking predicted-vs-actual correctness).
"""
from __future__ import annotations

import json
from typing import Any, Dict

from ..json_utils import extract_json
from ..trace import Trace

_SYSTEM_PROMPT = """You are the confidence-check component of a physics problem-solving agent.

Given a problem, its proposed solution, and the results of independent
logic/physics/math verification checks, estimate how confident you are
that the final answer is correct. Consider:
- Did any of the other checks fail? A failed check should push confidence down.
- Is the physics setup non-standard, ambiguous, or does it require
  assumptions that weren't stated?
- Is this the kind of problem where the verification tools have good
  coverage, or a case where nothing could fully verify the answer?

Respond with ONLY valid JSON, no commentary, no markdown fences:
{"confidence": <float between 0.0 and 1.0>, "rationale": "<one sentence>"}
"""


class ConfidenceCheck:
    name = "confidence"

    def __init__(self, llm_client, threshold: float = 0.6, max_retries: int = 1):
        self.llm = llm_client
        self.threshold = threshold
        self.max_retries = max_retries

    def run(self, trace: Trace) -> Dict[str, Any]:
        prior_checks = [
            {"check": name, "passed": name not in trace.checks_failed} for name in trace.checks_run
        ]
        user_content = json.dumps(
            {
                "problem": trace.problem_text,
                "initial_solution": trace.initial_solution,
                "prior_check_results": prior_checks,
            }
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw = ""
        last_err: Exception = ValueError("confidence check never ran")
        for _ in range(self.max_retries + 1):
            raw = self.llm.chat(messages)
            try:
                parsed = extract_json(raw)
                confidence = float(parsed.get("confidence"))
                confidence = max(0.0, min(1.0, confidence))
                rationale = str(parsed.get("rationale", ""))
                trace.final_confidence = confidence
                passed = confidence >= self.threshold
                details = f"confidence={confidence:.2f} (threshold={self.threshold}): {rationale}"
                return {"passed": passed, "details": details}
            except (ValueError, TypeError, json.JSONDecodeError) as e:
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

        # If confidence can't even be extracted, treat that as a
        # zero-confidence situation rather than leaving the field unset.
        trace.final_confidence = 0.0
        return {
            "passed": False,
            "details": f"Confidence check failed to parse model output: {last_err}",
        }
