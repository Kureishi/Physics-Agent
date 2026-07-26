"""
Task Planner (Stage 1).

Given a raw physics problem, produces:
  - domain_tags: 1-3 tags from a fixed taxonomy (used later for retrieval,
    memory tagging, and per-domain meta-learning)
  - subtasks: an ordered decomposition of the steps needed to solve it

Both come from a single LLM call constrained to return JSON. Local models
served through LM Studio don't always respect "JSON only" instructions
perfectly (extra prose, markdown fences, etc.), so parsing is defensive and
the planner retries once with a corrective follow-up message before giving up.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List

DOMAIN_TAXONOMY = [
    "kinematics",
    "dynamics",
    "energy",
    "momentum",
    "rotational-dynamics",
    "gravitation",
    "oscillations-waves",
    "thermodynamics",
    "electromagnetism",
    "optics",
    "fluid-mechanics",
    "special-relativity",
    "quantum-mechanics",
    "statistical-mechanics",
]

_DECOMPOSE_SYSTEM_PROMPT = f"""You are the task planner for a physics problem-solving agent.

Given a physics problem, do two things:
1. Classify it with 1-3 tags from this fixed taxonomy only: {DOMAIN_TAXONOMY}
2. Decompose it into an ordered list of 3-6 concrete subtasks needed to solve it
   (for example: "identify knowns and unknowns", "select the governing equation",
   "solve algebraically", "check units", "check limiting cases").

Respond with ONLY valid JSON, no commentary, no markdown code fences, in exactly
this shape:
{{"domain_tags": ["tag1", "tag2"], "subtasks": ["step 1", "step 2", "step 3"]}}
"""


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Best-effort JSON extraction. Strips markdown code fences and pulls the
    first {...} block out of the response, since local models sometimes
    wrap valid JSON in prose ("Sure, here's the JSON: ...") or fences
    despite being told not to.
    """
    stripped = text.strip()
    stripped = re.sub(r"^```(json)?", "", stripped, flags=re.IGNORECASE).strip()
    stripped = re.sub(r"```$", "", stripped).strip()
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in planner output: {text!r}")
    return json.loads(match.group(0))


class TaskPlanner:
    def __init__(self, llm_client, max_retries: int = 1):
        self.llm = llm_client
        self.max_retries = max_retries

    def decompose(self, problem_text: str) -> Dict[str, Any]:
        """
        Returns:
            {
                "domain_tags": List[str],
                "subtasks": List[str],
                "raw_response": str,
                "planning_time_ms": float,
            }
        Raises ValueError if the model still doesn't produce valid, usable
        JSON after `max_retries` corrective follow-ups.
        """
        messages = [
            {"role": "system", "content": _DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": problem_text},
        ]

        start = time.time()
        last_err: Exception = ValueError("planner never ran")
        raw = ""
        for _ in range(self.max_retries + 1):
            raw = self.llm.chat(messages)
            try:
                parsed = _extract_json(raw)
                domain_tags = [t for t in parsed.get("domain_tags", []) if isinstance(t, str)]
                subtasks = [s for s in parsed.get("subtasks", []) if isinstance(s, str)]
                if not subtasks:
                    raise ValueError("Planner returned zero subtasks")

                # Keep only tags in the known taxonomy; unknown tags are
                # dropped rather than silently corrupting downstream
                # domain-tagged memory/knowledge-graph lookups.
                domain_tags = [t for t in domain_tags if t in DOMAIN_TAXONOMY]

                elapsed_ms = (time.time() - start) * 1000
                return {
                    "domain_tags": domain_tags,
                    "subtasks": subtasks,
                    "raw_response": raw,
                    "planning_time_ms": elapsed_ms,
                }
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That response was not valid JSON in the required shape. "
                            "Reply again with ONLY the JSON object, nothing else."
                        ),
                    }
                )

        raise ValueError(
            f"Planner failed to produce valid JSON after {self.max_retries + 1} attempts: "
            f"{last_err}\nLast raw output: {raw!r}"
        )
