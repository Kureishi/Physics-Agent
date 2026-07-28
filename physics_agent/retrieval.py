"""
Retrieval (Stage 1) / Semantic Memory (Stage 5): a semantic-memory store
for physics formulas and concepts.

Schema per entry matches the semantic-memory design:
    {id, statement, conditions, confidence, provenance, tags, last_validated}

Retrieval is deliberately simple here — keyword-overlap scoring, no
embeddings — so Stage 1 can run end-to-end with zero extra ML dependencies
beyond the LLM call itself. `SemanticStore.retrieve` is the seam to swap in
an embedding-based version later without touching any calling code.

`record_outcome` (Stage 5) is what makes this genuinely a *memory* rather
than a static lookup table: a fact's confidence moves based on whether
solutions that leaned on it actually passed verification, instead of
staying frozen at its seed value forever.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class SemanticStore:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Semantic store seed file not found at {self.path}. "
                "See data/semantic_seed.json for the expected format."
            )
        with self.path.open("r", encoding="utf-8") as f:
            self.entries: List[Dict[str, Any]] = json.load(f)

    def retrieve(
        self, query: str, domain_tags: Optional[List[str]] = None, k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Score entries by token overlap with the query, with a bonus for
        matching domain tags (since a Stage-1-classified problem tag is a
        much stronger signal than raw keyword overlap). Returns the top k.
        """
        query_tokens = _tokenize(query)
        domain_tags = domain_tags or []
        scored = []
        for entry in self.entries:
            entry_tokens = _tokenize(entry["statement"] + " " + " ".join(entry.get("tags", [])))
            overlap = len(query_tokens & entry_tokens)
            tag_bonus = len(set(domain_tags) & set(entry.get("tags", []))) * 2
            score = overlap + tag_bonus
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored[:k]]

    def add(
        self,
        entry_id: str,
        statement: str,
        conditions: str,
        confidence: float,
        provenance: str,
        tags: List[str],
    ) -> None:
        """
        Adds a new fact to semantic memory and persists immediately. In
        later stages this is called by the outer/meta-learning loop when a
        new formula is derived and confirmed, not by the solving loop directly.
        """
        self.entries.append(
            {
                "id": entry_id,
                "statement": statement,
                "conditions": conditions,
                "confidence": confidence,
                "provenance": provenance,
                "tags": tags,
                "last_validated": time.time(),
            }
        )
        self._persist()

    def record_outcome(
        self, entry_id: str, success: bool, learning_rate: float = 0.1
    ) -> Optional[Dict[str, Any]]:
        """
        Stage 5: nudges an entry's confidence toward 1.0 on success or
        toward 0.0 on failure, via a simple exponential moving average
        (bounded to [0, 1] automatically since it's a convex combination of
        two values already in that range). Returns the updated entry, or
        None if entry_id wasn't found.

        This is deliberately a soft nudge rather than a hard overwrite: one
        solve's outcome is weak evidence about a fact that may have been
        used correctly for years before a single confused problem leaned
        on it in a case where something else was actually at fault.
        """
        for entry in self.entries:
            if entry["id"] == entry_id:
                target = 1.0 if success else 0.0
                entry["confidence"] = entry["confidence"] + learning_rate * (
                    target - entry["confidence"]
                )
                entry["last_validated"] = time.time()
                self._persist()
                return entry
        return None

    def _persist(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2)
