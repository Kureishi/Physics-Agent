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
    def __init__(self, llm_client, registry: ToolRegistry = None, max_retries: int = 1, tool_policy=None):
        self.llm = llm_client
        self.registry = registry or ToolRegistry()
        self.max_retries = max_retries
        # Stage 7: an optional learned policy that can narrow (never widen
        # or empty) which tools get offered for a domain, based on which
        # tools' presence in past first-attempts correlated with not
        # needing any correction. None preserves pre-Stage-7 behavior exactly.
        self.tool_policy = tool_policy

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
        return self.llm.chat(messages)

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
