"""
Trace schema — the single most important artifact in the system.

Every downstream stage (tool orchestration, multi-agent critique, memory
consolidation, self-correction, meta-learning) reads and writes traces
defined here. The full schema is defined now, at Stage 1, even though most
fields are only populated starting at later stages. This is deliberate:
retrofitting a trace schema after Stage 3-4 exist means losing the ability
to analyze anything solved before the retrofit.

Field ownership by stage (who fills each field in):
    Stage 1: problem_id, problem_text, timestamp, domain_tags, subtasks,
        retrieved_knowledge, planner_raw_response, planning_time_ms
    Stage 2: tool_calls, initial_solution, orchestration_time_ms
    Stage 3: checks_run, checks_failed, check_details, final_confidence
        (initial estimate)
    Stage 4 -- self-correction (this implementation): error_type,
        revision_count, revision_history, resolution_status, final_answer,
        time_to_solve_ms
    final_correct: intentionally left unset by every stage above. It would
        require comparison against an external ground-truth answer, which
        isn't implemented yet -- our own checks passing is evidence of
        self-consistency, not proof of correctness, and conflating the two
        would be a false claim baked into the trace log itself.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class ToolCall:
    tool: str
    input: str
    output: str
    latency_ms: float


@dataclass
class Trace:
    # --- identity ----------------------------------------------------------
    problem_id: str
    problem_text: str
    timestamp: float = field(default_factory=time.time)

    # --- Stage 1: planning + retrieval --------------------------------------
    domain_tags: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    retrieved_knowledge: List[Dict[str, Any]] = field(default_factory=list)
    planner_raw_response: Optional[str] = None
    planning_time_ms: Optional[float] = None

    # --- Stage 2: tool orchestration ----------------------------------------
    tool_calls: List[ToolCall] = field(default_factory=list)
    initial_solution: Optional[str] = None
    orchestration_time_ms: Optional[float] = None

    # --- Stage 3: self-evaluation / verification ----------------------------
    checks_run: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    check_details: List[Dict[str, Any]] = field(default_factory=list)

    # --- Stage 4: self-correction --------------------------------------------
    error_type: Optional[str] = None
    revision_count: int = 0
    revision_history: List[Dict[str, Any]] = field(default_factory=list)
    resolution_status: Optional[str] = None

    # --- final outcome (populated once a full solve pipeline exists) -------
    final_answer: Optional[str] = None
    final_confidence: Optional[float] = None
    final_correct: Optional[bool] = None
    time_to_solve_ms: Optional[float] = None

    @staticmethod
    def new(problem_text: str) -> "Trace":
        return Trace(problem_id=str(uuid.uuid4()), problem_text=problem_text)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Trace":
        d = dict(d)
        tool_calls_raw = d.get("tool_calls", []) or []
        d["tool_calls"] = [
            tc if isinstance(tc, ToolCall) else ToolCall(**tc) for tc in tool_calls_raw
        ]
        # Drop unknown keys defensively so old traces still load if the
        # schema gains fields later.
        known_fields = {f for f in Trace.__dataclass_fields__}
        d = {k: v for k, v in d.items() if k in known_fields}
        return Trace(**d)


class EpisodicMemory:
    """
    Append-only JSONL store for traces. This is the raw substrate for a
    later structured-memory stage — one line per solved problem, easy to
    stream, diff, or re-process as later stages add richer fields.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def write(self, trace: Trace) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trace.to_dict()) + "\n")

    def read_all(self) -> List[Trace]:
        traces = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                traces.append(Trace.from_dict(json.loads(line)))
        return traces

    def query_by_domain_tags(self, tags: List[str]) -> List[Trace]:
        """Stage 5: traces whose domain_tags overlap with `tags` at all --
        e.g. for a later meta-learning stage asking "how are we doing on
        oscillations-waves problems lately?" Deliberately a simple linear
        scan/tag-overlap check, consistent with SemanticStore.retrieve's
        keyword approach -- no embeddings, no external index, just enough
        to be useful at the scale a single local agent operates at."""
        tag_set = set(tags)
        return [t for t in self.read_all() if tag_set & set(t.domain_tags)]

    def query_by_resolution_status(self, status: str) -> List[Trace]:
        """Stage 5: e.g. query_by_resolution_status("unresolved_max_revisions")
        to find problems the agent never actually managed to self-correct --
        exactly the set a later curriculum/meta-learning stage would want
        to prioritize."""
        return [t for t in self.read_all() if t.resolution_status == status]

    def query_by_error_type(self, error_type: str) -> List[Trace]:
        """Stage 5: all traces where this error_type was the last one
        detected -- useful for spotting a recurring failure pattern across
        many different problems, not just within one problem's revision_history."""
        return [t for t in self.read_all() if t.error_type == error_type]

    def __len__(self) -> int:
        return len(self.read_all())
