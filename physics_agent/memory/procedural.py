"""
Procedural Memory (Stage 5).

Tracks reusable strategies: for a given (domain_tags, error_type) pairing,
which Stage 4 corrective strategy was applied, and how often it actually
resolved the check(s) that triggered it. This is the substrate a later
meta-learning stage needs to eventually retune error_taxonomy's fixed
priority ordering based on real outcomes -- e.g. "does
'rederive_physics_setup' actually resolve cross_method_disagreement most
of the time, or is there a better strategy for that combination?"

Schema per entry:
    {id, domain_tags, error_type, strategy, n_uses, n_successes,
     success_rate, last_used}

Storage is a single flat JSON file (dict keyed by a composite id), not
JSONL -- unlike episodic memory, entries here are updated in place (a
running success rate), not appended as an immutable log.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _key(domain_tags: List[str], error_type: str, strategy: str) -> str:
    # Sorted so tag order never creates duplicate entries for what's
    # really the same (domain, error_type, strategy) combination.
    domain_part = "+".join(sorted(domain_tags)) or "none"
    return f"{domain_part}::{error_type}::{strategy}"


class ProceduralMemory:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", encoding="utf-8") as f:
                self.entries: Dict[str, Dict[str, Any]] = json.load(f)
        else:
            self.entries = {}
            self._persist()

    def record_outcome(
        self, domain_tags: List[str], error_type: str, strategy: str, resolved: bool
    ) -> Dict[str, Any]:
        """
        Called once per revision round (from MemoryConsolidator): did
        applying `strategy` for `error_type` in this domain context
        actually resolve the check(s) that triggered it? Updates a running
        success rate and persists immediately.
        """
        key = _key(domain_tags, error_type, strategy)
        entry = self.entries.get(key)
        if entry is None:
            entry = {
                "id": key,
                "domain_tags": sorted(domain_tags),
                "error_type": error_type,
                "strategy": strategy,
                "n_uses": 0,
                "n_successes": 0,
                "success_rate": 0.0,
                "last_used": None,
            }
        entry["n_uses"] += 1
        if resolved:
            entry["n_successes"] += 1
        entry["success_rate"] = entry["n_successes"] / entry["n_uses"]
        entry["last_used"] = time.time()
        self.entries[key] = entry
        self._persist()
        return entry

    def get(self, domain_tags: List[str], error_type: str, strategy: str) -> Optional[Dict[str, Any]]:
        return self.entries.get(_key(domain_tags, error_type, strategy))

    def best_strategy_for(self, domain_tags: List[str], error_type: str) -> Optional[Dict[str, Any]]:
        """
        Returns the entry with the highest success_rate among strategies
        tried for this (domain, error_type) combination, requiring at
        least 3 uses to avoid acting on a single noisy sample. Returns
        None if there isn't enough data yet.

        Not used to override error_taxonomy's fixed strategy choice
        anywhere in this stage -- that override is a meta-learning-stage
        decision -- but the data is collected here specifically so that
        decision has something real to work from later.
        """
        candidates = [
            e
            for e in self.entries.values()
            if e["error_type"] == error_type
            and set(e["domain_tags"]) & set(domain_tags)
            and e["n_uses"] >= 3
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e["success_rate"])

    def all_entries(self) -> List[Dict[str, Any]]:
        return list(self.entries.values())

    def _persist(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2)
