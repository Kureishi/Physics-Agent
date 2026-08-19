"""
Tool Orchestrator (Stage 2).

Given a Stage-1 trace (problem text, domain tags, subtasks, retrieved
knowledge), this:

  1. Asks the LLM which tools to call and with what inputs, constrained to
     the tools relevant for this problem's domain (ToolRegistry.relevant_tools).
  2. Executes each tool call, capturing input/output/latency into
     trace.tool_calls. Tool FAILURES are captured as ToolCall entries too
     (with an error payload as the output) rather than raised -- a failed
     tool call is exactly the kind of signal Stage 3 (self-evaluation) and
     Stage 5 (self-correction) need to see, so it must survive in the trace,
     not crash the pipeline.
  3. Synthesizes an initial solution from the tool outputs via one more LLM
     call, written to trace.initial_solution.

trace.initial_solution is the "Initial Solution" box in the architecture
diagram. Everything from here on (self-evaluation, error detection,
revision) is a later stage operating on this trace.

Synthesis retries on an empty response (found via a real accumulated run,
not speculatively): the same "model returns empty / truncated-mid-sentence
with no error" failure mode ProblemGenerator's docstring documents in
detail for problem generation also happens during solving's synthesis
step, and for the same likely reason (a "thinking"-style model spending
its budget on an internal reasoning phase before any visible answer).
Before this fix, _synthesize_solution had no retry at all -- an empty
completion was accepted as-is, `trace.initial_solution` ended up `""`,
Logic Check correctly failed it ("No initial solution provided to
evaluate"), and Stage 4's `resynthesize` strategy (the correction Stage 4
picks for a Logic-only failure) called this same un-retried function
again -- with nothing different about the retry to make a second empty
response less likely. Measured directly against one real run: 13 of 19
unresolved-after-max-revisions traces had an empty initial_solution, and
several domains showed `resynthesize` sitting at a flat 0% success rate
across many uses -- not a verification bug, a generation gap that the
verification layer was correctly (but unproductively) catching over and
over. See _synthesize_solution below for the fix, which mirrors
ProblemGenerator's retry-with-lower-temperature approach directly.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from .json_utils import extract_json
from .tools.registry import ToolRegistry
from .trace import Trace, ToolCall

_TOOL_SELECTION_SYSTEM_PROMPT_TEMPLATE = """You are the tool-selection component of a physics problem-solving agent.

Given a physics problem, its subtasks, and physics facts already retrieved
from memory, decide which tools to call and with what inputs to make
progress toward a solution.

Available tools for this problem: {available_tools}

Tool input formats:
- symbolic_math: {{"expression": "<sympy-parseable equation, e.g. Eq(0.5*m*v**2, m*g*h)>", "solve_for": "<variable>", "substitutions": {{"var": value, ...}}}}
- simulation: {{"state_vars": [...], "derivatives": [...], "params": {{...}}, "initial_conditions": {{...}}, "t_span": [t0, t1]}}
- literature_search: {{"query": "...", "max_results": 3}}

You may call zero or more tools. Prefer symbolic_math when the problem has
a clean closed-form solution. Only use tools from the available list above.

Respond with ONLY valid JSON, no commentary, no markdown fences, in exactly
this shape:
{{"tool_calls": [{{"tool": "<name>", "input": {{...}}}}, ...]}}
If no tool call is useful, respond with {{"tool_calls": []}}.

If the user message includes a "revision_feedback" field, a previous
attempt at this problem failed verification for the reasons described
there -- choose tool calls that specifically address those issues rather
than repeating the same approach.
"""

_SYNTHESIS_SYSTEM_PROMPT = """You are the synthesis component of a physics problem-solving agent.
Given the original problem, its subtasks, and the results of any tool calls
made, write a clear, step-by-step initial solution, ending with the final
numeric or symbolic answer including units. If a tool call failed or
returned nothing useful, work around it using the retrieved physics facts
and your own reasoning, and note explicitly where you had to do so.

If the user message includes a "revision_feedback" field, a previous
attempt failed verification for the reasons described there -- make sure
your solution specifically corrects those issues, not just restates the
same reasoning.
"""


class ToolOrchestrator:
    def __init__(
        self,
        llm_client,
        registry: ToolRegistry = None,
        max_retries: int = 1,
        tool_policy=None,
        synthesis_retry_temperature: float = 0.1,
    ):
        self.llm = llm_client
        self.registry = registry or ToolRegistry()
        self.max_retries = max_retries
        # Stage 7: an optional learned policy that can narrow (never widen
        # or empty) which tools get offered for a domain, based on which
        # tools' presence in past first-attempts correlated with not
        # needing any correction. None preserves pre-Stage-7 behavior exactly.
        self.tool_policy = tool_policy
        # Used for every synthesis retry after the first (see module
        # docstring): solving already defaults to a low temperature (0.2),
        # so this only needs to nudge lower still -- unlike
        # ProblemGenerator's retry_temperature (0.3), which is a much
        # bigger drop from its deliberately-high 0.9 generation-diversity
        # default.
        self.synthesis_retry_temperature = synthesis_retry_temperature

    # -- tool selection -------------------------------------------------

    def _select_tool_calls(self, trace: Trace, feedback: Optional[str] = None) -> List[Dict[str, Any]]:
        available = self.registry.relevant_tools(trace.domain_tags)
        if self.tool_policy is not None:
            available = self.tool_policy.filter_tools(trace.domain_tags, available)
        system_prompt = _TOOL_SELECTION_SYSTEM_PROMPT_TEMPLATE.format(available_tools=available)

        payload = {
            "problem": trace.problem_text,
            "subtasks": trace.subtasks,
            "retrieved_knowledge": [
                {"statement": k["statement"], "conditions": k["conditions"]}
                for k in trace.retrieved_knowledge
            ],
        }
        if feedback:
            payload["revision_feedback"] = feedback
        user_content = json.dumps(payload)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        last_err: Exception = ValueError("tool selection never ran")
        raw = ""
        for _ in range(self.max_retries + 1):
            raw = self.llm.chat(messages)
            try:
                parsed = extract_json(raw)
                tool_calls = parsed.get("tool_calls", [])
                if not isinstance(tool_calls, list):
                    raise ValueError("'tool_calls' must be a list")
                # Drop hallucinated/unavailable tool names rather than
                # crashing on them -- the model may name a tool that
                # wasn't offered.
                valid_calls = [
                    tc for tc in tool_calls if isinstance(tc, dict) and tc.get("tool") in available
                ]
                return valid_calls
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
            f"Tool selection failed to produce valid JSON after {self.max_retries + 1} "
            f"attempts: {last_err}\nLast raw output: {raw!r}"
        )

    # -- tool execution ---------------------------------------------------

    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[ToolCall]:
        executed = []
        for call in tool_calls:
            tool_name = call["tool"]
            tool_input = call.get("input", {}) or {}
            tool = self.registry.get(tool_name)

            start = time.time()
            try:
                output = tool.run(tool_input)
                output_str = json.dumps(output)
            except Exception as e:
                # Capture the failure in the trace; never let a bad tool
                # call take down the whole pipeline.
                output_str = json.dumps({"error": str(e)})
            latency_ms = (time.time() - start) * 1000

            executed.append(
                ToolCall(
                    tool=tool_name,
                    input=json.dumps(tool_input),
                    output=output_str,
                    latency_ms=latency_ms,
                )
            )
        return executed

    # -- synthesis ----------------------------------------------------------

    def _synthesize_solution(self, trace: Trace, feedback: Optional[str] = None) -> str:
        """
        Retries on an empty response, same failure mode and same fix
        shape as ProblemGenerator.generate() (see that module's docstring
        for the full "why lower temperature, not higher max_tokens"
        reasoning) -- see this module's own docstring for why this retry
        was missing here specifically and what it cost in practice.

        Raises RuntimeError if every attempt comes back empty. This is a
        deliberate change from the old silent-empty-string behavior:
        letting an empty trace.initial_solution flow forward just meant
        Logic Check would (correctly) fail it, `resynthesize` would call
        this same function again with nothing different about the retry,
        and three revisions later the trace ends up unresolved anyway --
        just slower, and with the underlying cause (no LLM output at all)
        undistinguished from a genuine reasoning failure in the trace.
        Raising here instead surfaces the real cause immediately.
        cli.run() does not currently catch this, matching how any other
        LLM-call failure (e.g. LM Studio not running) already propagates
        uncaught today; problem_set_cli.py's own per-problem try/except
        already isolates a batch run from a single problem raising here.
        """
        tool_results_summary = [
            {"tool": tc.tool, "input": tc.input, "output": tc.output} for tc in trace.tool_calls
        ]
        payload = {
            "problem": trace.problem_text,
            "subtasks": trace.subtasks,
            "retrieved_knowledge": [k["statement"] for k in trace.retrieved_knowledge],
            "tool_results": tool_results_summary,
        }
        if feedback:
            payload["revision_feedback"] = feedback
        user_content = json.dumps(payload)
        messages = [
            {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            # First attempt uses the client's own configured temperature
            # (solving defaults to 0.2); every retry after that uses the
            # lower synthesis_retry_temperature instead, for the same
            # reason ProblemGenerator's retry_temperature exists: an
            # empty/degenerate completion is a classic high-temperature
            # sampling failure, and retrying at the same temperature that
            # just failed has no particular reason to succeed differently.
            call_temperature = None if attempt == 0 else self.synthesis_retry_temperature
            try:
                raw = self.llm.chat(messages, temperature=call_temperature)
            except Exception as e:
                last_err = e
                raw = ""

            if raw and raw.strip():
                return raw

            last_err = last_err or ValueError("empty response")
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That response was empty. Provide the full written solution this "
                        "time, ending with the final numeric or symbolic answer."
                    ),
                }
            )

        raise RuntimeError(
            f"Synthesis produced an empty response after {self.max_retries + 1} attempt(s): "
            f"{last_err}. Likely the model spent its full token budget on an internal "
            "reasoning phase before producing any visible answer -- see ProblemGenerator's "
            "docstring for the same failure mode found during problem generation."
        )

    # -- public entry points ---------------------------------------------------

    def run(self, trace: Trace, feedback: Optional[str] = None) -> Trace:
        """
        Mutates and returns `trace`: populates trace.tool_calls,
        trace.initial_solution, and trace.orchestration_time_ms.

        `feedback`, if given, is Stage 4's mechanism for asking for a
        revised attempt: it's threaded into both the tool-selection and
        synthesis prompts as "revision_feedback", so a from-scratch
        re-derivation can specifically address what failed verification
        last time, rather than the LLM having no idea a previous attempt
        existed.
        """
        start = time.time()
        tool_calls = self._select_tool_calls(trace, feedback=feedback)
        trace.tool_calls = self._execute_tool_calls(tool_calls)
        trace.initial_solution = self._synthesize_solution(trace, feedback=feedback)
        trace.orchestration_time_ms = (time.time() - start) * 1000
        return trace

    def resynthesize(self, trace: Trace, feedback: Optional[str] = None) -> str:
        """
        Stage 4's "resynthesize" correction strategy: re-run only the
        synthesis step, leaving trace.tool_calls untouched. Used when the
        self-eval failure was about the reasoning/write-up (Logic Check),
        not about the tool calls or physics setup themselves -- redoing
        tool calls in that case would be wasted work.
        """
        solution = self._synthesize_solution(trace, feedback=feedback)
        trace.initial_solution = solution
        return solution

    def escalate_with_literature_search(self, trace: Trace, feedback: Optional[str] = None) -> Trace:
        """
        Stage 4's "escalate_verification" correction strategy: used when
        confidence is low but no specific check failed, so there's no
        concrete error to fix. Rather than blindly redoing work that
        already passed every specific check, pull in one more independent
        signal (a literature search) and let synthesis incorporate it.
        """
        tool = self.registry.get("literature_search")
        start = time.time()
        try:
            output = tool.run({"query": trace.problem_text, "max_results": 2})
            output_str = json.dumps(output)
        except Exception as e:
            output_str = json.dumps({"error": str(e)})
        latency_ms = (time.time() - start) * 1000

        trace.tool_calls.append(
            ToolCall(
                tool="literature_search",
                input=json.dumps({"query": trace.problem_text, "max_results": 2}),
                output=output_str,
                latency_ms=latency_ms,
            )
        )
        trace.initial_solution = self._synthesize_solution(trace, feedback=feedback)
        return trace
