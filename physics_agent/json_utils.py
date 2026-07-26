"""
Shared helper for defensively extracting JSON from LLM output.

Local models served through LM Studio don't always follow "output only
JSON" instructions perfectly -- they wrap it in prose, markdown fences, or
both. Both the planner (Stage 1) and the tool orchestrator (Stage 2) need
this same defensive extraction, so it lives in one place.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict


def extract_json(text: str) -> Dict[str, Any]:
    """
    Strips markdown code fences and pulls the first {...} block out of the
    response. Raises ValueError if no JSON object can be found/parsed, so
    callers can decide whether to retry or fail loudly.
    """
    stripped = text.strip()
    stripped = re.sub(r"^```(json)?", "", stripped, flags=re.IGNORECASE).strip()
    stripped = re.sub(r"```$", "", stripped).strip()
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    return json.loads(match.group(0))
