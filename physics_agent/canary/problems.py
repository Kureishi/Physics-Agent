"""
Canary problems -- the fixed set of human-verified problems referenced in
the feedback-loop design doc's Safety Rails section:

    "A small, fixed set of problems with known, human-verified correct
    answers, run periodically... Compare the pipeline's own checks against
    the known answer, not just against themselves."

These are deliberately NOT drawn from the regular problem sets
(data/problem_sets/*.json). Those exist to exercise the pipeline broadly;
canaries exist to audit it, so their answers are hand-computed and checked
into data/canary_problems.json rather than left for the pipeline itself to
judge -- the entire point is a ground truth the system did not produce.

Each entry has one scalar expected_value with a relative_tolerance, not a
free-form expected answer string. This is a deliberate scope limit: it
lets grading (see grading.py) be a plain numeric comparison rather than
another judgment call an LLM has to make -- exactly the kind of
self-referential grading this feature exists to avoid leaning on. The
trade-off is that a canary can only exercise problems with one clean
scalar answer; that's an acceptable limit for a small audit set, not a
statement that every physics problem reduces to one number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union


@dataclass
class CanaryProblem:
    id: str
    domain_hint: str
    problem_text: str
    expected_value: float
    relative_tolerance: float
    units: str
    verified_by: str


def load_canary_problems(path: Union[str, Path]) -> List[CanaryProblem]:
    """
    Raises FileNotFoundError / json.JSONDecodeError on a missing or
    malformed file rather than silently returning an empty set -- a canary
    suite that quietly runs zero canaries because its data file went
    missing would defeat the entire point (see runner.py, which treats an
    empty canary set as a loud error, not a clean pass).
    """
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return [
        CanaryProblem(
            id=entry["id"],
            domain_hint=entry["domain_hint"],
            problem_text=entry["problem_text"],
            expected_value=float(entry["expected_value"]),
            relative_tolerance=float(entry.get("relative_tolerance", 0.02)),
            units=entry.get("units", ""),
            verified_by=entry.get("verified_by", ""),
        )
        for entry in raw
    ]
