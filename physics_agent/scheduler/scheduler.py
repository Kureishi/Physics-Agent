"""
Scheduler -- the background process itself, closing the gap the design doc
named directly: "the single biggest lever: an orchestration layer that
decides when to solve, review, and practice, instead of a person
deciding... This alone converts 'a person runs the training workflow'
into 'the training workflow runs, and a person checks in on it.'"

Every existing CLI in this project (cli.py, curriculum_cli.py,
meta_report.py, problem_set_cli.py) is a manual trigger sitting on top of
an otherwise self-contained loop -- a person still has to decide *when* to
run each one. This module is that missing decision layer: one `run_cycle()`
call does whatever combination of the following is actually warranted
right now, using the exact same underlying functions those CLIs already
call (physics_agent.cli.run, meta_learning.report.build_report,
CurriculumRunner.run_round) rather than reimplementing any of them:

  1. Solve -- if the problem queue (see queue.py) has anything pending,
     solve the next one via cli.run(), exactly as problem_set_cli.py does
     for a whole batch.
  2. Review -- once scheduler_review_every_n_solves new solves have
     accumulated since the last review, run build_report() (the same
     report meta_report.py prints) and log a summary of what it found.
  3. Practice -- once weak_areas()'s top signal's weight clears
     scheduler_curriculum_weight_threshold, AND at least
     scheduler_curriculum_min_cycles_between_rounds cycles have passed
     since the last curriculum round, run one CurriculumRunner round
     targeting it.
  4. Grow -- once at least scheduler_growth_min_cycles_between_rounds
     cycles have passed since the last growth round, check
     knowledge_growth.find_candidate_facts() and commit any that clear
     its bar via propose_and_add(). A heavier, rarer action than the
     other three (it writes new persistent semantic-memory entries, not
     just a report or a practice round), hence its own, larger cooldown.

Review and practice are decoupled from whether a solve happened THIS
cycle: review triggers off an accumulated solve count (so an idle queue
correctly produces no review -- there's nothing new to review), but
practice triggers off weak_areas()'s standing signal, which can already be
elevated from historical data alone. A persistently under-resourced queue
should still get periodic practice rounds even if nothing new is flowing
through it.

Each action taken is logged as a separate Decision with its own rationale
-- "logs the decision, not just the outcome" (design doc) -- for the same
auditability reason inspect_trace_cli.py mattered for individual traces.
A cycle that does nothing (empty queue, review/practice not due) still
logs a single "idle" decision, so an empty decision log is
distinguishable from "the scheduler hasn't run" rather than looking the
same as "the scheduler ran and correctly did nothing."

This module intentionally does NOT daemonize, background itself, or
manage a process lifecycle -- see scheduler_cli.py for `--loop`, a plain
sleep-and-repeat wrapper meant to be run under systemd/cron/nohup/tmux,
whichever process supervisor a given deployment already uses. Rebuilding
that supervision layer here would be scope creep this project doesn't
need.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .. import cli
from ..config import Config
from ..curriculum.curriculum_runner import CurriculumRunner
from ..knowledge_graph.graph import KnowledgeGraph
from ..memory.error_memory import ErrorMemory
from ..memory.procedural import ProceduralMemory
from ..meta_learning.curriculum_signals import weak_areas
from ..meta_learning.knowledge_growth import ProposedFactsRegistry, find_candidate_facts, propose_and_add
from ..meta_learning.report import build_report
from ..retrieval import SemanticStore
from .queue import ProblemQueue
from .state import SchedulerState, load_state, save_state
from ..trace import EpisodicMemory


@dataclass
class Decision:
    timestamp: float
    action: str  # "solve" | "review" | "practice" | "grow" | "idle"
    reason: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Decision":
        return Decision(**d)


class DecisionLog:
    """Append-only JSONL store for scheduler decisions -- same pattern as
    CanaryLog/CurriculumLog: one line per decision, immutable history."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def write(self, decision: Decision) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(decision.to_dict()) + "\n")

    def read_all(self) -> List[Decision]:
        decisions = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    decisions.append(Decision.from_dict(json.loads(line)))
        return decisions


class Scheduler:
    def __init__(self, config: Optional[Config] = None, dry_run: bool = False):
        self.config = config or Config()
        self.dry_run = dry_run
        self.queue = ProblemQueue(self.config.scheduler_queue_path)
        self.decision_log = DecisionLog(self.config.scheduler_log_path)
        self.state = load_state(self.config.scheduler_state_path)

    def run_cycle(self) -> List[Decision]:
        """
        One tick of the loop: at most one solve, plus whichever of
        review/practice are due, each logged as its own Decision. Always
        logs at least one Decision (an explicit "idle" one when nothing
        else happened this cycle), and always persists updated state
        before returning, so a crash between cycles loses at most the
        in-progress cycle's counters, not the whole cadence history.
        """
        self.state.total_cycles += 1
        self.state.cycles_since_last_curriculum += 1
        self.state.cycles_since_last_growth += 1

        decisions: List[Decision] = []

        solve_decision = self._maybe_solve()
        if solve_decision is not None:
            decisions.append(solve_decision)

        review_decision = self._maybe_review()
        if review_decision is not None:
            decisions.append(review_decision)

        practice_decision = self._maybe_practice()
        if practice_decision is not None:
            decisions.append(practice_decision)

        grow_decision = self._maybe_grow()
        if grow_decision is not None:
            decisions.append(grow_decision)

        if not decisions:
            decisions.append(
                Decision(
                    timestamp=time.time(),
                    action="idle",
                    reason=(
                        "queue is empty, a review isn't due yet "
                        f"({self.state.solves_since_last_review}/"
                        f"{self.config.scheduler_review_every_n_solves} solves since last review), "
                        "no weak-area signal currently clears the practice threshold "
                        "(or a curriculum round ran too recently), "
                        "and no repeated-expression signal clears the growth threshold "
                        "(or a growth round ran too recently)"
                    ),
                    details={},
                )
            )

        for decision in decisions:
            self.decision_log.write(decision)

        save_state(self.config.scheduler_state_path, self.state)
        return decisions

    def run_loop(
        self,
        interval_seconds: float,
        max_cycles: Optional[int] = None,
        sleep_fn=time.sleep,
    ) -> None:
        """Repeats run_cycle() forever (or max_cycles times, for tests /
        bounded runs), sleeping interval_seconds between cycles. See
        scheduler_cli.py for the process-level wrapper around this."""
        cycles_run = 0
        while max_cycles is None or cycles_run < max_cycles:
            self.run_cycle()
            cycles_run += 1
            if max_cycles is None or cycles_run < max_cycles:
                sleep_fn(interval_seconds)

    # -- individual decisions --------------------------------------------

    def _maybe_solve(self) -> Optional[Decision]:
        problem = self.queue.pop_next()
        if problem is None:
            return None

        trace = cli.run(
            problem["problem_text"],
            dry_run=self.dry_run,
            config=self.config,
            source="scheduler_queue",
        )

        self.state.total_solves += 1
        self.state.solves_since_last_review += 1

        return Decision(
            timestamp=time.time(),
            action="solve",
            reason=f"queue had a pending problem ('{problem.get('id', '?')}'); solved it",
            details={
                "problem_id": problem.get("id"),
                "trace_id": trace.problem_id,
                "resolution_status": trace.resolution_status,
                "domain_tags": trace.domain_tags,
                "queue_remaining": len(self.queue),
            },
        )

    def _maybe_review(self) -> Optional[Decision]:
        if self.state.solves_since_last_review < self.config.scheduler_review_every_n_solves:
            return None

        store = SemanticStore(self.config.semantic_store_path)
        graph = KnowledgeGraph(self.config.knowledge_graph_path, store)
        episodic = EpisodicMemory(self.config.episodic_memory_path)
        procedural = ProceduralMemory(self.config.procedural_memory_path)
        error_memory = ErrorMemory(self.config.error_memory_path)
        report = build_report(episodic, procedural, error_memory, graph)

        n_anomalies = len(report["check_value_anomalies"]["flags"])
        n_escalations = len(report["escalations"])
        n_declining = len(report["declining_strategies"])

        self.state.total_reviews += 1
        self.state.solves_since_last_review = 0

        return Decision(
            timestamp=time.time(),
            action="review",
            reason=(
                f"{self.config.scheduler_review_every_n_solves} solves accumulated since the last "
                f"review; ran a fresh meta-learning report over {report['n_traces']} total trace(s) "
                f"({n_anomalies} check-value anomaly flag(s), {n_escalations} escalation flag(s), "
                f"{n_declining} declining strategy(ies))"
            ),
            details={
                "n_traces": report["n_traces"],
                "n_anomaly_flags": n_anomalies,
                "n_escalations": n_escalations,
                "n_declining_strategies": n_declining,
            },
        )

    def _maybe_practice(self) -> Optional[Decision]:
        if self.state.cycles_since_last_curriculum < self.config.scheduler_curriculum_min_cycles_between_rounds:
            return None

        store = SemanticStore(self.config.semantic_store_path)
        graph = KnowledgeGraph(self.config.knowledge_graph_path, store)
        episodic = EpisodicMemory(self.config.episodic_memory_path)
        error_memory = ErrorMemory(self.config.error_memory_path)
        signals = weak_areas(episodic, error_memory, graph, limit=1)

        if not signals or signals[0]["weight"] < self.config.scheduler_curriculum_weight_threshold:
            return None

        top_signal = signals[0]
        runner = CurriculumRunner(self.config, dry_run=self.dry_run)
        results = runner.run_round(n_problems=self.config.scheduler_curriculum_n_problems)

        self.state.total_curriculum_rounds += 1
        self.state.cycles_since_last_curriculum = 0

        return Decision(
            timestamp=time.time(),
            action="practice",
            reason=(
                f"top weak-area signal (\"{top_signal['reason']}\", weight={top_signal['weight']}) "
                f"cleared the practice threshold ({self.config.scheduler_curriculum_weight_threshold}); "
                f"ran a curriculum round targeting it"
            ),
            details={
                "targeted_signal": top_signal,
                "n_problems_generated": len(results),
                "resulting_trace_ids": [r.resulting_trace_id for r in results],
            },
        )

    def _maybe_grow(self) -> Optional[Decision]:
        if self.state.cycles_since_last_growth < self.config.scheduler_growth_min_cycles_between_rounds:
            return None

        episodic = EpisodicMemory(self.config.episodic_memory_path)
        candidates = find_candidate_facts(episodic)

        if not candidates:
            return None

        semantic = SemanticStore(self.config.semantic_store_path)
        registry = ProposedFactsRegistry(self.config.proposed_facts_registry_path)
        added = propose_and_add(semantic, registry, candidates)

        self.state.cycles_since_last_growth = 0

        if not added:
            # Candidates existed but every one had already been proposed
            # in an earlier round -- still resets the cooldown (a real
            # check happened) but isn't its own loggable "grow" event
            # distinct from that earlier round.
            return None

        self.state.total_growth_rounds += 1

        top = added[0]
        return Decision(
            timestamp=time.time(),
            action="grow",
            reason=(
                f"{len(added)} repeated-expression signal(s) cleared the growth threshold "
                f"(top: \"{top['statement']}\" used successfully across {top['n_observations']} "
                "independently-solved problems); proposed as new low-confidence semantic fact(s)"
            ),
            details={
                "n_facts_added": len(added),
                "added": added,
            },
        )
