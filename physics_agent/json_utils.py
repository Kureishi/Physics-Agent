"""
Shared helper for defensively extracting JSON from LLM output.

Local models served through LM Studio don't always follow "output only
JSON" instructions perfectly -- they wrap it in prose, markdown fences, or
both. This same defensive extraction is shared by every JSON-producing
component: the planner (Stage 1), the tool orchestrator (Stage 2), all
three LLM-based self-evaluation checks (Stage 3), and the curriculum's
problem generator (Stage 8).

It also sanitizes stray backslashes before parsing -- see
_sanitize_invalid_escapes below. This was found to be a real, reproducible
failure in practice: physics content naturally reaches for LaTeX
(\\text{...}, \\mu_0, \\le, \\pi, \\cdot, \\times), and a model writing that
directly inside a JSON string value without escaping the backslash itself
either crashes json.loads outright (\\l, \\m, \\p, \\c aren't valid JSON
escapes) or -- worse -- silently corrupts the content (\\t IS a valid JSON
escape for tab, so "\\text{mm}" parses "successfully" as a tab character
followed by the garbled literal text "ext{mm}", with no error raised at all).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_ESCAPE_RE = re.compile(r"\\(.)")

# Backslash sequences left untouched -- unambiguous, and \u must be left
# alone since json.loads needs to see it to consume the following 4 hex
# digits as a unicode escape.
_ALWAYS_VALID_ESCAPES = {'"', "\\", "/", "u"}

# Deliberately NOT special-casing \b \f \n \r \t as "usually genuine
# escapes, unless followed by more letters": that heuristic sounds
# reasonable but doesn't actually work here, since a deliberate \n
# separating two sentences is just as often followed by more lowercase
# prose as a LaTeX command's next letter is (e.g. "\nEnd of problem" vs
# "\nabla") -- there's no reliable way to tell those apart from the
# character alone. Given this system's fields are short, single-paragraph
# prose (problem_text, rationale, issues) rather than long structured
# documents, treating ALL of b/f/n/r/t as "escape it" rather than "trust
# it" is the safer default: the cost is a genuine deliberate \n/\t/\r
# surviving as a literal 2-character sequence instead of an actual control
# character (cosmetic), versus the alternative of silently corrupting or
# crashing on the far more common case of LaTeX physics notation.


def _sanitize_invalid_escapes(text: str) -> str:
    def _replace(match: "re.Match[str]") -> str:
        escaped_char = match.group(1)
        if escaped_char in _ALWAYS_VALID_ESCAPES:
            return match.group(0)
        return "\\\\" + escaped_char

    return _ESCAPE_RE.sub(_replace, text)


def extract_json(text: str) -> Dict[str, Any]:
    """
    Strips markdown code fences, pulls the first {...} block out of the
    response, and defensively escapes stray backslashes (see
    _sanitize_invalid_escapes) before parsing. Raises ValueError if no
    JSON object can be found/parsed, so callers can decide whether to
    retry or fail loudly.
    """
    stripped = text.strip()
    stripped = re.sub(r"^```(json)?", "", stripped, flags=re.IGNORECASE).strip()
    stripped = re.sub(r"```$", "", stripped).strip()
    match = _JSON_BLOCK_RE.search(stripped)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    sanitized = _sanitize_invalid_escapes(match.group(0))
    return json.loads(sanitized)
