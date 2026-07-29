"""
Self-Evaluation Pipeline (Stage 3).

Runs Logic, Physics, Math, and Confidence checks in order over a Stage-2
trace (problem + initial_solution + tool_calls), populating
trace.checks_run, trace.checks_failed, and trace.check_details.

Order matters: ConfidenceCheck runs last since it takes the other checks'
pass/fail results as one of its own inputs.

Every check is run inside a try/except here -- a single check crashing
(e.g. a bug in a check itself, or an LLM call raising instead of returning)
must not take down the whole pipeline. It's recorded as a failed check with
the exception message as its detail, which is itself useful signal for
Stage 5's self-correction / Stage 7's meta-learning to notice a
mis-behaving check.
"""
from __future__ import annotations

from typing import List, Optional

from .confidence_check import ConfidenceCheck
from .logic_check import LogicCheck
from .math_check import MathCheck
from .physics_check import PhysicsCheck
from ..trace import Trace


class SelfEvaluationPipeline:
    def __init__(
        self,
        llm_client=None,
        confidence_threshold: float = 0.6,
        checks: Optional[List] = None,
        knowledge_graph=None,
        verification_depth_policy=None,
    ):
        """
        `checks`, if given, overrides the default check list entirely
        (mainly for testing pipeline-level behavior, e.g. crash handling,
        without needing a real or mock LLM client wired through every check).

        `knowledge_graph` (Stage 6), if given, is threaded into PhysicsCheck
        so it can run its deterministic assumption-validity sub-check
        alongside cross-tool agreement and the LLM critique.

        `verification_depth_policy` (Stage 7), if given, is threaded into
        ConfidenceCheck so it can raise the effective pass threshold for
        domains where confidence has historically run ahead of what this
        system's own outcomes justify.
        """
        self.checks = checks if checks is not None else [
            LogicCheck(llm_client),
            PhysicsCheck(llm_client, knowledge_graph=knowledge_graph),
            MathCheck(),
            ConfidenceCheck(
                llm_client, threshold=confidence_threshold, threshold_policy=verification_depth_policy
            ),
        ]

    def run(self, trace: Trace) -> Trace:
        for check in self.checks:
            trace.checks_run.append(check.name)
            try:
                result = check.run(trace)
            except Exception as e:
                result = {"passed": False, "details": f"Check raised an unexpected exception: {e}"}

            trace.check_details.append(
                {"check": check.name, "passed": result["passed"], "details": result["details"]}
            )
            if not result["passed"]:
                trace.checks_failed.append(check.name)

        return trace
