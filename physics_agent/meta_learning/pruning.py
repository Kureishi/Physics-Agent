"""
Procedural-Memory Pruning Signal (Stage 7).

Flags -- does not delete or modify -- procedural-memory entries whose
success rate has stayed low despite enough attempts to trust the number:
"strategies with declining success rates get flagged for revision" from
the design doc.

Deliberately read-only: this never mutates memory/procedural.json, so
Stage 5's MemoryConsolidator remains the only writer to that file. Keeping
"record what happened" (Stage 5) and "flag what might need to change"
(this module) as separate writers/readers keeps it auditable which
component touched the data and why.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..memory.procedural import ProceduralMemory

MIN_USES_TO_FLAG = 5
LOW_SUCCESS_RATE_THRESHOLD = 0.34  # roughly "fails more often than it works"


def flag_declining_strategies(
    procedural_memory: ProceduralMemory,
    min_uses: int = MIN_USES_TO_FLAG,
    threshold: float = LOW_SUCCESS_RATE_THRESHOLD,
) -> List[Dict[str, Any]]:
    return sorted(
        (
            entry
            for entry in procedural_memory.all_entries()
            if entry["n_uses"] >= min_uses and entry["success_rate"] < threshold
        ),
        key=lambda e: e["success_rate"],
    )
