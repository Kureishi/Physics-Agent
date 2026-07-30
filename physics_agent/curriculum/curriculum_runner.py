"""
Curriculum Runner (Stage 8).

Closes the loop the original design doc described: takes Stage 7's ranked
weak-area signals, generates new practice problems targeting them, and
solves those problems through the exact same Stage 1-7 pipeline every
other problem goes through (physics_agent.cli.run) -- so curriculum
problems get self-evaluated, self-corrected, and consolidated into memory
exactly like any other problem, and their outcomes feed right back into
the same statistics that produced the weak-area signal in the first place.

For each targeted signal, measures the exact underlying metric that
produced it (error_memory frequency, count of unresolved traces in a
domain, or knowledge-graph cluster average confidence) before and after
solving the generated practice problem. This is a genuinely measured
comparison, not a guaranteed-positive one -- practicing a domain doesn't
always move the specific metric that was flagged (the generated problem
might not even touch the exact knowledge-graph node in question, for
example), and this module reports that honestly.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .. import cli
from ..config import Config
from ..knowledge_graph.graph import KnowledgeGraph
from ..llm_client import LLMClient, MockLLMClient
from ..memory.error_memory import ErrorMemory
from ..meta_learning.curriculum_signals import weak_areas
from ..retrieval import SemanticStore
from ..tools.literature import LiteratureSearchTool
from ..trace import EpisodicMemory
from .problem_generator import ProblemGenerator


@dataclass
class CurriculumRoundResult:
    round_id: str
    timestamp: float
    targeted_signal: Dict[str, Any]
    generated_problem_text: str
    target_concepts: List[str]
    literature_context: Optional[str]
    resulting_trace_id: str
    resolution_status: Optional[str]
    final_confidence: Optional[float]
    metric_description: str
    metric_before: Optional[float]
    metric_after: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CurriculumRoundResult":
        return CurriculumRoundResult(**d)


class CurriculumLog:
    """
    Append-only JSONL store for curriculum ROUND summaries -- mirrors
    EpisodicMemory's pattern, but one entry per curriculum round, not per
    problem (the underlying generated problems are already in episodic
    memory, written by cli.run() exactly like any other problem).
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def write(self, result: CurriculumRoundResult) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def read_all(self) -> List[Dict[str, Any]]:
        entries = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries


def _measure_signal(
    signal: Dict[str, Any],
    error_memory: ErrorMemory,
    knowledge_graph: KnowledgeGraph,
    episodic_memory: EpisodicMemory,
) -> Tuple[Optional[float], str]:
    """
    Returns (metric_value, description) for a weak-area signal, read from
    the exact same store that produced it -- so a before/after comparison
    measures the thing that was actually flagged, not a proxy for it.
    """
    source = signal["source"]

    if source == "error_memory":
        error_type = signal["error_type"]
        entry = error_memory.get(error_type, signal["domain_tags"])
        value = float(entry["frequency"]) if entry else 0.0
        return value, f"error_memory frequency for '{error_type}' in {signal['domain_tags']}"

    if source == "episodic_memory":
        tag = signal["domain_tags"][0] if signal["domain_tags"] else None
        unresolved = episodic_memory.query_by_resolution_status("unresolved_max_revisions")
        count = float(len([t for t in unresolved if tag in t.domain_tags])) if tag else 0.0
        return count, f"count of unresolved traces tagged '{tag}'"

    if source == "knowledge_graph":
        node_ids = signal.get("node_ids", [])
        confidences = []
        for node_id in node_ids:
            node = knowledge_graph.get_node(node_id)
            if node:
                confidences.append(node["confidence"])
        avg = (sum(confidences) / len(confidences)) if confidences else None
        return avg, f"average confidence across knowledge graph cluster {node_ids}"

    return None, f"unrecognized signal source '{source}'"


class CurriculumRunner:
    def __init__(self, config: Config, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run

    def _build_generator(self) -> ProblemGenerator:
        llm = (
            MockLLMClient()
            if self.dry_run
            else LLMClient(
                base_url=self.config.lm_studio_base_url,
                api_key=self.config.lm_studio_api_key,
                model=self.config.lm_studio_model,
            )
        )
        # Real literature search only in real runs -- dry runs shouldn't
        # depend on network access, consistent with how the rest of the
        # dry-run path avoids it.
        literature_tool = None if self.dry_run else LiteratureSearchTool()
        return ProblemGenerator(llm, literature_tool=literature_tool)

    def run_round(self, n_problems: int = 1) -> List[CurriculumRoundResult]:
        store = SemanticStore(self.config.semantic_store_path)
        knowledge_graph = KnowledgeGraph(self.config.knowledge_graph_path, store)
        episodic = EpisodicMemory(self.config.episodic_memory_path)
        error_memory = ErrorMemory(self.config.error_memory_path)
        curriculum_log = CurriculumLog(self.config.curriculum_log_path)

        signals = weak_areas(episodic, error_memory, knowledge_graph, limit=n_problems)
        if not signals:
            return []

        generator = self._build_generator()
        results: List[CurriculumRoundResult] = []

        for signal in signals:
            metric_before, metric_description = _measure_signal(
                signal, error_memory, knowledge_graph, episodic
            )

            try:
                generated = generator.generate(signal)
            except ValueError:
                # This one signal's generation failed -- skip it, but keep
                # going on the others rather than losing the whole round.
                continue

            trace = cli.run(
                generated["problem_text"],
                dry_run=self.dry_run,
                config=self.config,
                source="curriculum",
                curriculum_target=signal,
            )

            # Re-load stores fresh for the "after" measurement, since
            # cli.run() wrote new data to disk through its own instances.
            store_after = SemanticStore(self.config.semantic_store_path)
            knowledge_graph_after = KnowledgeGraph(self.config.knowledge_graph_path, store_after)
            episodic_after = EpisodicMemory(self.config.episodic_memory_path)
            error_memory_after = ErrorMemory(self.config.error_memory_path)
            metric_after, _ = _measure_signal(
                signal, error_memory_after, knowledge_graph_after, episodic_after
            )

            result = CurriculumRoundResult(
                round_id=trace.problem_id,
                timestamp=time.time(),
                targeted_signal=signal,
                generated_problem_text=generated["problem_text"],
                target_concepts=generated["target_concepts"],
                literature_context=generated["literature_context"],
                resulting_trace_id=trace.problem_id,
                resolution_status=trace.resolution_status,
                final_confidence=trace.final_confidence,
                metric_description=metric_description,
                metric_before=metric_before,
                metric_after=metric_after,
            )
            curriculum_log.write(result)
            results.append(result)

        return results
