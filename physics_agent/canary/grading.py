"""
Canary grading -- deterministic numeric comparison of a solved trace
against a canary's hand-verified expected_value.

Two design choices worth calling out:

1. Grading prefers the deterministic tool output (symbolic_math's
   `solutions_numeric`, simulation's `final_state`) over parsing
   `trace.final_answer` prose, and only falls back to prose extraction
   when no usable tool output exists. This whole feature exists to check
   the pipeline against something independent of its own self-consistency
   -- grading it primarily off the LLM-synthesized prose answer would
   partially defeat that, since a paraphrase step sits between the actual
   computed number and the text being graded. See extract_candidate_values.

2. Grading is a plain "does any candidate number fall within tolerance"
   check, not an LLM judgment call. This is deliberately dumb: a canary's
   job is to be a source of ground truth the rest of the system can be
   checked against, so it shouldn't itself depend on the kind of judgment
   it exists to audit.

Known limitation (documented rather than silently hidden): numeric
extraction has no unit awareness. If a canary's expected_value is in Pa
and the pipeline reports the equivalent value in kPa, extraction will
pull out a number 1000x too small and the canary will read as a failure
that isn't really one. The canary problem set mitigates this by asking
explicitly for a stated unit in each problem_text, but this is a
mitigation, not a guarantee -- a real failure surfaced this way is worth
a human glance at `matched_value` / `extraction_source` before assuming
the pipeline is actually wrong.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..trace import Trace

# Unicode superscript digits/signs, as produced by LLMs writing scientific
# notation like "5.466×10⁻¹⁴" instead of "5.466e-14".
_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")

# "<base> × 10 <exponent>" / "<base> x 10^<exponent>", exponent either
# unicode superscript digits or plain ASCII digits.
_SCI_UNICODE_RE = re.compile(
    r"(?P<base>[-+]?\d+(?:\.\d+)?)\s*[×xX]\s*10\s*(?:\^)?\s*"
    r"(?P<exp>[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+|[-+]?\d+)"
)

# Negative lookbehind for a letter excludes the trailing digit in a
# subscripted variable name (v1, m2, x0) from being read as its own
# number -- those are identifiers, not values, and without this a
# problem's own restated inputs ("m1 = 3 kg") could spuriously "match" an
# unrelated expected_value that happens to equal the subscript digit.
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _normalize_scientific_unicode(text: str) -> str:
    def _replace(match: "re.Match[str]") -> str:
        base = match.group("base")
        exponent = match.group("exp").translate(_SUPERSCRIPT_MAP)
        return f"{base}e{exponent}"

    return _SCI_UNICODE_RE.sub(_replace, text)


def extract_numbers(text: str) -> List[float]:
    """
    Pulls every plausible numeric literal out of free text, handling plain
    decimals, e-notation (1.23e-4), and both ASCII (1.23 x 10^-4) and
    unicode (1.23×10⁻⁴) scientific notation. Returns candidates in the
    order found; callers decide which (if any) count as a match.
    """
    if not text:
        return []
    normalized = _normalize_scientific_unicode(text)
    numbers = []
    for match in _NUMBER_RE.finditer(normalized):
        try:
            numbers.append(float(match.group(0)))
        except ValueError:
            continue
    return numbers


def extract_candidate_values(trace: Trace) -> Tuple[List[float], str]:
    """
    Returns (candidate_values, source) where source is "tool_output" or
    "prose_fallback". Walks tool_calls in reverse (most recent first,
    consistent with EpisodicMemory/MathCheck's "last relevant call wins"
    convention elsewhere) looking for a symbolic_math or simulation call
    with parseable, non-error output; the first one found wins. Falls back
    to extracting numbers from trace.final_answer only if no tool call
    produced anything usable.
    """
    for tc in reversed(trace.tool_calls):
        if tc.tool not in ("symbolic_math", "simulation"):
            continue
        try:
            output = json.loads(tc.output)
        except json.JSONDecodeError:
            continue
        if "error" in output:
            continue

        values: List[float] = []
        if tc.tool == "symbolic_math":
            for v in output.get("solutions_numeric", []) or []:
                if v is not None:
                    values.append(float(v))
            if not values:
                # solutions_numeric can be all-None for a solution SymPy
                # left in symbolic (non-numeric) form; fall back to
                # parsing the string forms in that case.
                for s in output.get("solutions", []) or []:
                    values.extend(extract_numbers(str(s)))
        elif tc.tool == "simulation":
            final_state = output.get("final_state", {}) or {}
            for v in final_state.values():
                try:
                    values.append(float(v))
                except (TypeError, ValueError):
                    continue

        if values:
            return values, "tool_output"

    return extract_numbers(trace.final_answer or ""), "prose_fallback"


def best_match(
    candidates: List[float], expected_value: float, relative_tolerance: float
) -> Tuple[bool, Optional[float]]:
    """
    True/matched_value if any candidate is within relative_tolerance of
    expected_value (absolute tolerance fallback when expected_value is
    ~0, to avoid a divide-by-zero degenerate case -- none of the current
    canary set needs this, but a future canary near zero shouldn't crash
    grading). When multiple candidates match, returns the closest one.
    """
    if not candidates:
        return False, None

    if abs(expected_value) < 1e-12:
        matches = [c for c in candidates if abs(c) < 1e-9]
    else:
        matches = [
            c for c in candidates
            if abs(c - expected_value) / abs(expected_value) <= relative_tolerance
        ]

    if not matches:
        return False, None

    closest = min(matches, key=lambda c: abs(c - expected_value))
    return True, closest


@dataclass
class GradingResult:
    answer_correct: bool
    matched_value: Optional[float]
    extraction_source: str
    n_candidates: int


def grade_trace(
    trace: Trace, expected_value: float, relative_tolerance: float
) -> GradingResult:
    candidates, source = extract_candidate_values(trace)
    matched, matched_value = best_match(candidates, expected_value, relative_tolerance)
    return GradingResult(
        answer_correct=matched,
        matched_value=matched_value,
        extraction_source=source,
        n_candidates=len(candidates),
    )
