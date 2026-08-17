"""
Scheduler State.

Persists just enough counters across process restarts for the scheduling
cadence -- "review every N solves," "no more than one curriculum round
every M cycles" -- to survive a restart rather than silently resetting to
zero, which would either force an immediate review/round right after
restart or (worse) push the next one further out than intended.

This is deliberately NOT a history. Full history already lives elsewhere:
every action the scheduler takes is logged to scheduler_log.jsonl (see
scheduler.py's DecisionLog), and anything actually solved becomes a normal
episodic trace exactly like a manually-run problem. This file is only ever
the current cadence counters -- small, overwritten in place, never
appended to.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union


@dataclass
class SchedulerState:
    total_cycles: int = 0
    total_solves: int = 0
    total_reviews: int = 0
    total_curriculum_rounds: int = 0
    total_growth_rounds: int = 0
    solves_since_last_review: int = 0
    cycles_since_last_curriculum: int = 0
    cycles_since_last_growth: int = 0

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return SchedulerState(**d)


def load_state(path: Union[str, Path]) -> SchedulerState:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return SchedulerState()
    with p.open("r", encoding="utf-8") as f:
        return SchedulerState.from_dict(json.load(f))


def save_state(path: Union[str, Path], state: SchedulerState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)
