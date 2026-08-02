"""
LM Studio client.

LM Studio runs a local OpenAI-compatible server (Developer tab -> Start
Server; default http://localhost:1234/v1). Because it's OpenAI-compatible,
we use the official `openai` python package pointed at that base_url rather
than hand-rolling HTTP calls — this also means swapping to a different
OpenAI-compatible backend later (vLLM, an actual OpenAI key, etc.) is a
one-line config change, not a rewrite.

`timeout` and `max_tokens` exist specifically because of a real failure
mode found running this against different local models: one model
(a "thinking"/reasoning-tuned variant) appeared to hang indefinitely with
no output at all. Chat completions here are non-streaming -- nothing is
returned until the ENTIRE response, including any hidden chain-of-thought,
is complete -- so a model that reasons at length before emitting anything,
or one that simply never emits a stop token in a given quantization, looks
identical to "stuck forever" with no way to recover, since nothing in the
system could time out or cut it off. Both parameters are mitigations for
that, not a guaranteed fix for any specific model's behavior:
  - `timeout` bounds the wait on the client side. Whatever the model is
    doing, a request that runs longer than this raises a clear, catchable
    exception instead of hanging the whole process -- which, combined
    with the retry/skip handling already in TaskPlanner, ToolOrchestrator,
    the self-eval checks, and ProblemGenerator, turns "hangs forever" into
    "fails cleanly and gets skipped."
  - `max_tokens` bounds the response length on the server side, so a model
    stuck in a repetition loop (or one whose reasoning trace would
    otherwise run past any reasonable wait) gets cut off rather than
    running until it exhausts its context window. Note this is a genuine
    trade-off, not free: a model that legitimately needs a long reasoning
    trace before it can produce its actual answer may get truncated
    mid-thought and never produce a parseable response at all -- which
    still fails cleanly (the existing JSON-parsing retry path handles a
    truncated/incomplete response the same as any other unparseable one),
    but if you're using a model like that, raising `max_tokens` (and
    likely `timeout` alongside it) is the right knob to turn, at the cost
    of longer waits.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from openai import OpenAI

DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_TOKENS = 2048


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat completion endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        model: str = "local-model",
        temperature: float = 0.2,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
    ):
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        if effective_max_tokens is not None:
            kwargs["max_tokens"] = effective_max_tokens

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


class MockLLMClient:
    """
    Drop-in stand-in for LLMClient. Used for:
      - development/tests without LM Studio running
      - CI, where a local model server isn't available

    Matches LLMClient's interface exactly so calling code never needs to
    know which one it has.

    The pipeline now makes three distinct kinds of calls (planner
    decomposition, tool selection, solution synthesis), each with its own
    system prompt and expected response shape. `chat` recognizes which one
    it's serving by a marker substring unique to each system prompt, and
    returns a matching schema-valid default -- unless a caller-supplied
    canned response matches first.
    """

    DEFAULT_PLANNER_RESPONSE = (
        '{"domain_tags": ["dynamics", "energy"], '
        '"subtasks": ["Identify knowns and unknowns", '
        '"Choose the governing equation(s)", '
        '"Solve algebraically for the target quantity", '
        '"Check units and limiting cases"]}'
    )

    DEFAULT_TOOL_SELECTION_RESPONSE = (
        '{"tool_calls": [{"tool": "symbolic_math", "input": '
        '{"expression": "Eq(m*g*h, 0.5*m*v**2)", "solve_for": "v", '
        '"substitutions": {"m": 2, "g": 9.8, "h": 5}}}]}'
    )

    DEFAULT_SYNTHESIS_RESPONSE = (
        "Using conservation of energy (m*g*h = 0.5*m*v^2), the mass cancels "
        "and v = sqrt(2*g*h). The symbolic math tool solved this and returned "
        "the numeric result -- see the final answer in tool_results above."
    )

    DEFAULT_LOGIC_RESPONSE = '{"passed": true, "issues": []}'
    DEFAULT_PHYSICS_RESPONSE = '{"passed": true, "issues": []}'
    DEFAULT_CONFIDENCE_RESPONSE = '{"confidence": 0.85, "rationale": "Checks passed and the setup is standard."}'

    DEFAULT_CURRICULUM_RESPONSE = (
        '{"problem_text": "A 3 kg block starts at rest at the top of a 4 m '
        'tall frictionless incline. Find its speed at the bottom.", '
        '"target_concepts": ["energy conservation"], '
        '"rationale": "exercises the targeted weak area using a standard energy-conservation setup"}'
    )

    # Marker substrings unique to each system prompt, used to pick the
    # right default when nothing in `canned` matches.
    _TOOL_SELECTION_MARKER = "tool-selection component"
    _SYNTHESIS_MARKER = "synthesis component"
    _LOGIC_CHECK_MARKER = "logic-check component"
    _PHYSICS_CHECK_MARKER = "physics-check component"
    _CONFIDENCE_CHECK_MARKER = "confidence-check component"
    _CURRICULUM_MARKER = "problem generator"

    def __init__(self, canned_responses: Optional[Dict[str, str]] = None):
        # canned_responses: maps a substring -> the response to return when
        # that substring appears anywhere in the accumulated conversation
        # (system + user + any retry messages), so tests can target
        # specific problems or specific pipeline stages without needing a
        # real model.
        self.canned = canned_responses or {}
        self.calls: List[List[Dict[str, str]]] = []

    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        self.calls.append(messages)
        full_text = "\n".join(m["content"] for m in messages)

        for key, response in self.canned.items():
            if key in full_text:
                return response

        if self._TOOL_SELECTION_MARKER in full_text:
            return self.DEFAULT_TOOL_SELECTION_RESPONSE
        if self._SYNTHESIS_MARKER in full_text:
            return self.DEFAULT_SYNTHESIS_RESPONSE
        if self._LOGIC_CHECK_MARKER in full_text:
            return self.DEFAULT_LOGIC_RESPONSE
        if self._PHYSICS_CHECK_MARKER in full_text:
            return self.DEFAULT_PHYSICS_RESPONSE
        if self._CONFIDENCE_CHECK_MARKER in full_text:
            return self.DEFAULT_CONFIDENCE_RESPONSE
        if self._CURRICULUM_MARKER in full_text:
            return self.DEFAULT_CURRICULUM_RESPONSE
        return self.DEFAULT_PLANNER_RESPONSE
