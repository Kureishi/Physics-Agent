"""
Problem Queue -- what the scheduler's "solve" decision pulls from.

Deliberately the same JSON shape as data/problem_sets/*.json (a plain list
of {id, domain_hint, problem_text}), so the two ways the design doc
described filling it both just work with no adapter:
  - self-generated: `python -m physics_agent.generate_problem_set_cli
    --n-per-domain 5 --out data/problem_sets/scheduler_queue.json` (or
    --append onto an existing queue file)
  - externally supplied: hand-write or otherwise produce a JSON file in
    the same shape and point scheduler_queue_path at it

A literal FIFO: pop_next() removes and returns the first item, persisting
the shortened list immediately, so an item leaves the queue the moment
it's handed to the scheduler. This means the queue file itself does not
double as a record of what was processed -- that record already exists
independently, in the resulting episodic trace (via cli.run) and in the
scheduler's own decision log (see scheduler.py) -- so there's no need for
this module to also track consumed/pending status internally.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ProblemQueue:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write([])

    def _read(self) -> List[Dict[str, Any]]:
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, items: List[Dict[str, Any]]) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)

    def __len__(self) -> int:
        return len(self._read())

    def peek_all(self) -> List[Dict[str, Any]]:
        return self._read()

    def pop_next(self) -> Optional[Dict[str, Any]]:
        items = self._read()
        if not items:
            return None
        problem = items.pop(0)
        self._write(items)
        return problem

    def push(self, problem: Dict[str, Any]) -> None:
        items = self._read()
        items.append(problem)
        self._write(items)

    def extend(self, problems: List[Dict[str, Any]]) -> None:
        items = self._read()
        items.extend(problems)
        self._write(items)
