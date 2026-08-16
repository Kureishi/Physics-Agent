"""
Canary Runner -- Safety Rails, "Ground-truth canary problems" (feedback-loop
design doc, section 8/Safety Rails).

The MathCheck bug is the concrete motivating case: MathCheck flagged 190 of
199 direct-evaluation results as math failures, even though the underlying
answers were correct. Every other signal in this system checks the
pipeline against *itself* (does a check agree with another check, does a
strategy's success rate hold up) -- nothing checked a check against an
answer known to be right independent of the pipeline. That bug ran for an
entire accumulated history before anyone noticed, silently corrupting
confidence scores and strategy statistics along the way. A canary run
would have caught it in one pass: every canary whose tool call fell into
that direct-evaluation branch would have failed consistently and visibly.

This module solves each canary problem through the exact same pipeline
every other problem goes through (physics_agent.cli.run -- same convention
CurriculumRunner uses), then grades the *checks*, not just the answer:
given ground truth, did the pipeline's own self-eval agree with reality?

Four possible verdicts per canary (see CanaryResult.verdict):
    correct_and_passed    -- answer right, checks passed. Healthy.
    correct_but_flagged   -- answer right, checks failed anyway. A false
                              alarm -- exactly the MathCheck bug's shape.
                              Wastes revisions and corrupts statistics that
                              read "checks_failed" as evidence of an error.
    incorrect_and_flagged -- answer wrong, checks caught it. Healthy --
                              this is what the self-eval pipeline is for.
    incorrect_but_passed  -- answer wrong, checks passed anyway. The
                              dangerous case: a wrong answer shipped with
                              high confidence and nothing to flag it.
    unmeasurable          -- no numeric candidate could be extracted from
                              the trace at all (see grading.py); this is a
                              grading-pipeline gap, not evidence either way
                              about the solve itself.

Consistent with check_value.py's own stated philosophy: this module
produces the verdicts and logs them. Deciding what to *do* about a
recurring correct_but_flagged or incorrect_but_passed pattern (retrain a
check, escalate to a person, block an autonomous cycle) is left to a human
reviewing canary_cli's report or a future escalation-path feature -- this
is explicitly scoped to detection, not automated response.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .. import cli
from ..config import Config
from ..trace import Trace
from .grading import grade_trace
from .problems import CanaryProblem, load_canary_problems


@dataclass
class CanaryResult:
    canary_id: str
    timestamp: float
    domain_hint: str
    problem_text: str
    expected_value: float
    units: str
    matched_value: Optional[float]
    extraction_source: str
    n_candidates: int
    answer_correct: bool
    checks_failed: List[str]
    resolution_status: Optional[str]
    final_confidence: Optional[float]
    verdict: str
    trace_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CanaryResult":
        return CanaryResult(**d)


def _classify(answer_correct: bool, n_candidates: int, checks_failed: List[str]) -> str:
    if n_candidates == 0:
        return "unmeasurable"
    checks_passed = not checks_failed
    if answer_correct and checks_passed:
        return "correct_and_passed"
    if answer_correct and not checks_passed:
        return "correct_but_flagged"
    if not answer_correct and checks_passed:
        return "incorrect_but_passed"
    return "incorrect_and_flagged"


class CanaryLog:
    """
    Append-only JSONL store for canary run results -- mirrors
    EpisodicMemory/CurriculumLog's pattern, one entry per canary problem
    per run (so the same canary run repeatedly accumulates a history a
    future drift check can read back, the way check_value.py's report
    would if it were run over time).
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def write(self, result: CanaryResult) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def read_all(self) -> List[CanaryResult]:
        results = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(CanaryResult.from_dict(json.loads(line)))
        return results


class CanaryRunner:
    def __init__(self, config: Config, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run

    def run_all(self, problems: Optional[List[CanaryProblem]] = None) -> List[CanaryResult]:
        """
        Solves every canary problem through the full Stage 1-7 pipeline
        and grades each resulting trace. Raises ValueError on an empty
        canary set rather than returning an empty list -- a canary suite
        that silently runs nothing must not be mistaken for one that ran
        and found no problems (see problems.py's docstring for the same
        reasoning).
        """
        if problems is None:
            problems = load_canary_problems(self.config.canary_problems_path)
        if not problems:
            raise ValueError(
                "Canary problem set is empty -- refusing to report a clean run "
                "with zero canaries actually checked."
            )

        log = CanaryLog(self.config.canary_log_path)
        results: List[CanaryResult] = []

        for problem in problems:
            trace = cli.run(problem.problem_text, dry_run=self.dry_run, config=self.config)
            result = self._grade(problem, trace)
            log.write(result)
            results.append(result)

        return results

    def _grade(self, problem: CanaryProblem, trace: Trace) -> CanaryResult:
        grading = grade_trace(trace, problem.expected_value, problem.relative_tolerance)
        verdict = _classify(grading.answer_correct, grading.n_candidates, trace.checks_failed)

        return CanaryResult(
            canary_id=problem.id,
            timestamp=time.time(),
            domain_hint=problem.domain_hint,
            problem_text=problem.problem_text,
            expected_value=problem.expected_value,
            units=problem.units,
            matched_value=grading.matched_value,
            extraction_source=grading.extraction_source,
            n_candidates=grading.n_candidates,
            answer_correct=grading.answer_correct,
            checks_failed=list(trace.checks_failed),
            resolution_status=trace.resolution_status,
            final_confidence=trace.final_confidence,
            verdict=verdict,
            trace_id=trace.problem_id,
        )


def summarize(results: List[CanaryResult]) -> Dict[str, int]:
    """Counts of each verdict across a run -- the shape check_value.py and
    curriculum/benchmark.py both use for their own summaries."""
    counts = {
        "correct_and_passed": 0,
        "correct_but_flagged": 0,
        "incorrect_but_passed": 0,
        "incorrect_and_flagged": 0,
        "unmeasurable": 0,
    }
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    return counts


def latest_result_per_canary(log_entries: List[CanaryResult]) -> Dict[str, CanaryResult]:
    """Most recent logged result per canary_id, by timestamp -- used to
    compare a fresh run against history for drift (see canary_cli.py)."""
    latest: Dict[str, CanaryResult] = {}
    for entry in log_entries:
        current = latest.get(entry.canary_id)
        if current is None or entry.timestamp > current.timestamp:
            latest[entry.canary_id] = entry
    return latest
