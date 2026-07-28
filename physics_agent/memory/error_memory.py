"""
Error Memory (Stage 5).

A catalog of past mistakes: signature, root cause, the fix that was
applied, whether it resolved the issue, and how often that exact signature
recurs. This is what lets a recurring failure pattern become visible across
*many different problems* -- Stage 4's error_taxonomy classifies a single
problem's failure; this is where those classifications accumulate into
"this keeps happening," which is the trigger a later meta-learning stage
needs to decide a tool or heuristic itself needs fixing, not just this one
instance of it.

Schema per entry:
    {id, signature, error_type, domain_tags, root_cause, fix_applied,
     resolved, frequency, first_seen, last_seen}

A "signature" here is (error_type, domain_tags) -- coarse on purpose. A
finer-grained signature (e.g. including specific check_details text) would
fragment into near-unique entries and never accumulate frequency; the goal
is to notice *patterns*, not log every distinct sentence a check ever produced.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Union


def _signature(error_type: str, domain_tags: List[str]) -> str:
    domain_part = "+".join(sorted(domain_tags)) or "none"
    return f"{error_type}::{domain_part}"


class ErrorMemory:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", encoding="utf-8") as f:
                self.entries: Dict[str, Dict[str, Any]] = json.load(f)
        else:
            self.entries = {}
            self._persist()

    def record(
        self,
        error_type: str,
        domain_tags: List[str],
        root_cause: str,
        fix_applied: str,
        resolved: bool,
    ) -> Dict[str, Any]:
        sig = _signature(error_type, domain_tags)
        entry = self.entries.get(sig)
        now = time.time()
        if entry is None:
            entry = {
                "id": sig,
                "signature": sig,
                "error_type": error_type,
                "domain_tags": sorted(domain_tags),
                "root_cause": root_cause,
                "fix_applied": fix_applied,
                "resolved": resolved,
                "frequency": 0,
                "first_seen": now,
                "last_seen": now,
            }
        entry["frequency"] += 1
        # Keep the most recent root cause / fix description rather than
        # the first -- later occurrences' details are usually the more
        # relevant ones for "is this still a problem."
        entry["root_cause"] = root_cause
        entry["fix_applied"] = fix_applied
        entry["resolved"] = resolved
        entry["last_seen"] = now
        self.entries[sig] = entry
        self._persist()
        return entry

    def most_frequent(self, limit: int = 5) -> List[Dict[str, Any]]:
        """The recurring-pattern view: what keeps going wrong, ranked by
        how often it's happened -- exactly what a meta-learning /
        curriculum stage would scan first."""
        return sorted(self.entries.values(), key=lambda e: e["frequency"], reverse=True)[:limit]

    def get(self, error_type: str, domain_tags: List[str]) -> Dict[str, Any]:
        return self.entries.get(_signature(error_type, domain_tags))

    def all_entries(self) -> List[Dict[str, Any]]:
        return list(self.entries.values())

    def _persist(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2)
