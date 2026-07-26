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
    """

    DEFAULT_RESPONSE = (
        '{"domain_tags": ["dynamics", "energy"], '
        '"subtasks": ["Identify knowns and unknowns", '
        '"Choose the governing equation(s)", '
        '"Solve algebraically for the target quantity", '
        '"Check units and limiting cases"]}'
    )

    def __init__(self, canned_responses: Optional[Dict[str, str]] = None):
        # canned_responses: maps a substring of the user prompt -> the
        # response to return when that substring is present, so tests can
        # target specific problems without needing a real model.
        self.canned = canned_responses or {}
        self.calls: List[List[Dict[str, str]]] = []

    def chat(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        self.calls.append(messages)
        user_content = messages[-1]["content"]
        for key, response in self.canned.items():
            if key in user_content:
                return response
        return self.DEFAULT_RESPONSE
