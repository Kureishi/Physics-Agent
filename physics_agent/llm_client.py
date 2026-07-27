"""
LM Studio client.

LM Studio runs a local OpenAI-compatible server (Developer tab -> Start
Server; default http://localhost:1234/v1). Because it's OpenAI-compatible,
we use the official `openai` python package pointed at that base_url rather
than hand-rolling HTTP calls — this also means swapping to a different
OpenAI-compatible backend later (vLLM, an actual OpenAI key, etc.) is a
one-line config change, not a rewrite.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from openai import OpenAI


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat completion endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        model: str = "local-model",
        temperature: float = 0.2,
    ):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature

    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
        )
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

    # Marker substrings unique to each system prompt, used to pick the
    # right default when nothing in `canned` matches.
    _TOOL_SELECTION_MARKER = "tool-selection component"
    _SYNTHESIS_MARKER = "synthesis component"
    _LOGIC_CHECK_MARKER = "logic-check component"
    _PHYSICS_CHECK_MARKER = "physics-check component"
    _CONFIDENCE_CHECK_MARKER = "confidence-check component"

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
        return self.DEFAULT_PLANNER_RESPONSE
