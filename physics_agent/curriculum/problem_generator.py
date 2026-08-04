"""
Problem Generator (Stage 8, also used for general problem-set expansion).

Given a "signal" describing what to generate for -- either a real weak-area
signal from Stage 7's meta_learning.curriculum_signals.weak_areas, or a
generic domain-tag + reason pair when you just want more problems in a
domain with no weakness data yet -- asks the LLM to write ONE new,
self-contained physics practice problem.

Optionally grounded in a real literature_search result for the signal's
domain -- "read literature" here means the generator sees a title and a
short excerpt from a real paper and may draw inspiration from its general
subject matter, not that it comprehends or reproduces the paper itself.
Copying source text verbatim is explicitly forbidden in the prompt,
consistent with this system's broader stance on not reproducing source
material (see LiteratureSearchTool's own docstring on the same point).

`generate()` also accepts an optional `avoid` list of previously generated
problem texts (or short summaries of them) for the same domain, so calling
it repeatedly in a loop -- as generate_problem_set_cli.py does -- doesn't
just produce N near-identical variations of the first idea a local model
reaches for.

Retry handling covers three distinct failure modes, all hit in practice
running this against real local models: the response coming back as
unparseable/wrongly-shaped JSON (handled since Stage 8's original
implementation); the underlying LLM call itself failing outright, e.g. a
local inference engine returning a raw server error (handled once Bug 4
was found); and -- found generating problems in bulk across many domains
in one run -- the model returning an EMPTY or truncated-mid-sentence
response with no error at all. That last one showed a telling pattern:
raising max_tokens from 2048 to 8192 didn't reduce how often it happened
(if anything, a run with the higher cap succeeded on fewer problems than
one with the default), which rules out simple token starvation as the
cause. The far more likely explanation is a "thinking"-style model
spending an unbounded, prompt-dependent amount of its budget on an
internal reasoning phase before ever starting the visible answer --
giving it a bigger cap doesn't fix that, since the model can just think
longer, not proportionally leave more room for the answer.

What actually helps a failure like that is different from what helps a
parsing failure: retrying with a LOWER temperature. High temperature
(this module defaults to 0.9 for generation diversity, deliberately
higher than solving's 0.2) is exactly the setting most likely to produce
a degenerate empty or early-stopped completion; a lower, more
conservative temperature on retry trades away some of that diversity
specifically to prioritize actually getting a usable response. Every
retry attempt after the first uses `retry_temperature` instead of
whatever the client's own default is.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..json_utils import extract_json
from ..tools.literature import LiteratureSearchTool

_SYSTEM_PROMPT = """You are the problem generator for a physics self-improvement agent.

Given domain tags and a reason for the request -- which may be a specific
weakness the agent has shown (a recurring error type, a pattern of
unresolved problems, a low-confidence concept cluster) or simply a request
for more general practice material in a domain -- write ONE new,
well-posed, self-contained physics practice problem.

Requirements:
- State all needed numeric values in the problem itself (no missing data).
- The problem must have a clear, well-defined final numeric or symbolic answer.
- Match difficulty typical of an introductory-to-intermediate physics course.
- Write all math in plain text, NOT LaTeX and NOT unicode math symbols --
  no backslash commands (\\text, \\mu, \\pi, \\times, \\cdot, \\le), no
  special unicode characters (rho, mu, <=, x). Use plain ASCII instead:
  "mu_0", "pi", "<=", "x", "rho", spelled-out units like "mm" or "m/s".
  This keeps your response reliably parseable as JSON.
- If literature context is provided, you may draw inspiration from its
  general subject matter, but you MUST write an original problem in your
  own words -- never copy phrases or sentences from the provided text.
- If a list of problems to avoid duplicating is provided, write something
  meaningfully different from all of them -- a different physical setup,
  not just different numbers plugged into the same scenario.

Respond with ONLY valid JSON, no commentary, no markdown fences:
{"problem_text": "...", "target_concepts": ["concept1", "concept2"], "rationale": "one sentence on what this exercises"}
"""


class ProblemGenerator:
    def __init__(
        self,
        llm_client,
        literature_tool: Optional[LiteratureSearchTool] = None,
        max_retries: int = 1,
        retry_temperature: float = 0.3,
    ):
        self.llm = llm_client
        self.literature_tool = literature_tool
        self.max_retries = max_retries
        # Used for every retry attempt after the first (see module
        # docstring): lower temperature trades diversity for reliability
        # specifically when the first, higher-temperature attempt
        # produced an empty, truncated, or otherwise unusable response.
        self.retry_temperature = retry_temperature

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

    def generate(
        self, signal: Dict[str, Any], avoid: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        `signal`: {"source", "domain_tags", "reason", "weight", ...} --
        either a real entry from weak_areas(), or a synthetic one built for
        general expansion (see generate_problem_set_cli.py).

        `avoid`: previously generated problem texts for this same domain,
        so repeated calls don't just restate the same scenario with
        different numbers.

        Returns:
            {"problem_text", "target_concepts", "rationale", "literature_context"}
        Raises ValueError if the model can't produce valid, usable JSON --
        or the underlying LLM call itself keeps failing -- after
        `max_retries` additional attempts.
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
        if avoid:
            # Truncate each to keep the prompt from growing unbounded over
            # a long generation run -- a short prefix is enough for the
            # model to recognize "don't repeat this scenario."
            user_payload["avoid_duplicating"] = [text[:200] for text in avoid]

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload)},
        ]

        raw = ""
        last_err: Exception = ValueError("problem generator never ran")
        for attempt in range(self.max_retries + 1):
            # First attempt uses the client's own configured temperature
            # (generate_problem_set_cli.py sets this high, for diversity);
            # every retry after that uses the lower, more conservative
            # retry_temperature instead, since a degenerate/empty response
            # is a classic high-temperature sampling failure and blindly
            # retrying at the same temperature that just failed has no
            # particular reason to succeed differently.
            call_temperature = None if attempt == 0 else self.retry_temperature

            try:
                raw = self.llm.chat(messages, temperature=call_temperature)
            except Exception as e:
                # The LLM call itself failed (network error, a local
                # inference engine returning a raw server error on this
                # particular generation, etc.) -- treat it the same as an
                # unparseable response: worth a retry, and if it keeps
                # happening, surfaced as the same kind of ValueError a
                # parsing failure would raise below.
                last_err = e
                continue

            if not raw or not raw.strip():
                # Distinguish this explicitly from "the model replied with
                # something, just not valid JSON" -- an empty response
                # never even reaches extract_json, and "No JSON object
                # found in model output: ''" is a much less useful error
                # message than naming what actually happened.
                last_err = ValueError(
                    "Model returned an empty response (no content at all) -- "
                    "likely spent its entire token budget on an internal "
                    "reasoning phase before producing any visible answer."
                )
                messages.append({"role": "assistant", "content": ""})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That response was empty. Reply again with ONLY the "
                            "JSON object, no other text before or after it."
                        ),
                    }
                )
                continue

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
