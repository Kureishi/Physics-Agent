"""
Confidence Check (Stage 3, extended in Stage 7) -- the "Confidence Check"
box in the self-evaluation pipeline.

Asks the LLM for a confidence estimate given the problem, the initial
solution, and the outcome of the Logic/Physics/Math checks that already
ran. Below the effective threshold, the check itself is marked failed --
not because anything was definitively wrong, but as a signal to escalate
(Stage 5: deeper verification, or human review) rather than silently
shipping a low-confidence answer.

This is also where trace.final_confidence gets its first value. Stage 4
self-correction may revise it after a correction pass; Stage 7's
VerificationDepthPolicy (if provided) is what actually calibrates whether
the threshold itself is trustworthy over time, by raising it for domains
where confidence has historically run ahead of what this system's own
outcomes justify.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

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

    def __init__(
        self,
        llm_client,
        threshold: float = 0.6,
        max_retries: int = 1,
        threshold_policy=None,
    ):
        self.llm = llm_client
        self.threshold = threshold
        self.max_retries = max_retries
        # Stage 7: an optional VerificationDepthPolicy that can raise (never
        # lower) the effective threshold for a domain based on historical
        # calibration. None preserves pre-Stage-7 behavior exactly.
        self.threshold_policy = threshold_policy

    def _effective_threshold(self, trace: Trace) -> float:
        if self.threshold_policy is None:
            return self.threshold
        return self.threshold_policy.recommended_confidence_threshold(trace.domain_tags, self.threshold)

    def run(self, trace: Trace) -> Dict[str, Any]:
        effective_threshold = self._effective_threshold(trace)
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
                passed = confidence >= effective_threshold
                threshold_note = (
                    f" [raised from {self.threshold} by verification-depth policy]"
                    if effective_threshold > self.threshold
                    else ""
                )
                details = (
                    f"confidence={confidence:.2f} (threshold={effective_threshold:.2f}"
                    f"{threshold_note}): {rationale}"
                )
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
