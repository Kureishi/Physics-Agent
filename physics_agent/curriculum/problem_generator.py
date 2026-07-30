"""
Problem Generator (Stage 8).

Given a weak-area signal (one entry from Stage 7's
meta_learning.curriculum_signals.weak_areas), asks the LLM to write ONE
new, self-contained physics practice problem targeting it.

Optionally grounded in a real literature_search result for the signal's
domain -- "read literature" here means the generator sees a title and a
short excerpt from a real paper and may draw inspiration from its general
subject matter, not that it comprehends or reproduces the paper itself.
Copying source text verbatim is explicitly forbidden in the prompt,
consistent with this system's broader stance on not reproducing source
material (see LiteratureSearchTool's own docstring on the same point).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..json_utils import extract_json
from ..tools.literature import LiteratureSearchTool

_SYSTEM_PROMPT = """You are the curriculum generator for a physics self-improvement agent.

Given domain tags and a description of a weakness the agent has shown (a
recurring error type, a pattern of unresolved problems, or a
low-confidence concept cluster), write ONE new, well-posed, self-contained
physics practice problem that would exercise this weak area.

Requirements:
- State all needed numeric values in the problem itself (no missing data).
- The problem must have a clear, well-defined final numeric or symbolic answer.
- Match difficulty typical of an introductory-to-intermediate physics course.
- If literature context is provided, you may draw inspiration from its
  general subject matter, but you MUST write an original problem in your
  own words -- never copy phrases or sentences from the provided text.

Respond with ONLY valid JSON, no commentary, no markdown fences:
{"problem_text": "...", "target_concepts": ["concept1", "concept2"], "rationale": "one sentence on how this exercises the weak area"}
"""


class ProblemGenerator:
    def __init__(
        self,
        llm_client,
        literature_tool: Optional[LiteratureSearchTool] = None,
        max_retries: int = 1,
    ):
        self.llm = llm_client
        self.literature_tool = literature_tool
        self.max_retries = max_retries

    def _literature_context(self, domain_tags: List[str]) -> Optional[str]:
        if self.literature_tool is None or not domain_tags:
            return None
        query = " ".join(domain_tags)
        try:
            result = self.literature_tool.run({"query": query, "max_results": 1})
        except Exception:
            return None
        results = result.get("results", [])
        if not results:
            return None
        top = results[0]
        return f"{top['title']}: {top['excerpt']}"

    def generate(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        `signal` is one entry from weak_areas():
            {"source", "domain_tags", "reason", "weight", ...}

        Returns:
            {"problem_text", "target_concepts", "rationale", "literature_context"}
        Raises ValueError if the model can't produce valid, usable JSON
        after `max_retries` corrective follow-ups.
        """
        domain_tags = signal.get("domain_tags", [])
        literature_context = self._literature_context(domain_tags)

        user_payload = {
            "domain_tags": domain_tags,
            "weakness_reason": signal.get("reason", ""),
            "signal_source": signal.get("source", ""),
        }
        if literature_context:
            user_payload["literature_context"] = literature_context

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload)},
        ]

        raw = ""
        last_err: Exception = ValueError("problem generator never ran")
        for _ in range(self.max_retries + 1):
            raw = self.llm.chat(messages)
            try:
                parsed = extract_json(raw)
                problem_text = parsed.get("problem_text")
                if not problem_text or not isinstance(problem_text, str):
                    raise ValueError("Generated response missing non-empty 'problem_text'")
                target_concepts = [c for c in parsed.get("target_concepts", []) if isinstance(c, str)]
                rationale = str(parsed.get("rationale", ""))
                return {
                    "problem_text": problem_text,
                    "target_concepts": target_concepts,
                    "rationale": rationale,
                    "literature_context": literature_context,
                }
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

        raise ValueError(
            f"Problem generation failed after {self.max_retries + 1} attempts: "
            f"{last_err}\nLast raw output: {raw!r}"
        )
