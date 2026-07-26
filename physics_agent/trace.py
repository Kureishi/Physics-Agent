"""
Trace schema — the single most important artifact in the system.

Every downstream stage (tool orchestration, multi-agent critique, memory
consolidation, self-correction, meta-learning) reads and writes traces
defined here. The full schema is defined now, at Stage 1, even though most
fields are only populated starting at later stages. This is deliberate:
retrofitting a trace schema after Stage 3-4 exist means losing the ability
to analyze anything solved before the retrofit.

Field ownership by stage (who fills each field in):
    Stage 1 (this implementation): problem_id, problem_text, timestamp,
        domain_tags, subtasks, retrieved_knowledge, planner_raw_response,
        planning_time_ms
    Stage 2 (tool orchestration):   tool_calls
    Stage 3 (self-evaluation):      checks_run, checks_failed
    Stage 5 (self-correction):      error_type, revision_count
    Final outcome (any stage):      final_answer, final_confidence,
        final_correct, time_to_solve_ms
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

    # --- Stage 2+: tool orchestration (empty until that stage is built) ----
    tool_calls: List[ToolCall] = field(default_factory=list)

    # --- Stage 3: self-evaluation / verification (empty until built) -------
    checks_run: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)

    # --- Stage 5: self-correction (empty until built) -----------------------
    error_type: Optional[str] = None
    revision_count: int = 0

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
    Append-only JSONL store for traces. This is the raw substrate for
    Stage 4 (structured memory) — one line per solved problem, easy to
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

    def __len__(self) -> int:
        return len(self.read_all())
