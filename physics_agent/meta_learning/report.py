"""
Meta-Learning Report (Stage 7).

Ties the reporting-only signals together into one call for a periodic
"how is the agent doing" review -- see physics_agent/meta_report.py for a
CLI entry point that prints this. Does not include the two signals that
actively change behavior (ToolSelectionPolicy, VerificationDepthPolicy),
since those are consulted live during solving (wired into cli.py), not
reviewed after the fact.
"""
from __future__ import annotations

from typing import Any, Dict

from .check_value import compute_check_value_report
from .curriculum_signals import weak_areas
from .pruning import flag_declining_strategies
from ..knowledge_graph.graph import KnowledgeGraph
from ..memory.error_memory import ErrorMemory
from ..memory.procedural import ProceduralMemory
from ..trace import EpisodicMemory


def build_report(
    episodic_memory: EpisodicMemory,
    procedural_memory: ProceduralMemory,
    error_memory: ErrorMemory,
    knowledge_graph: KnowledgeGraph,
) -> Dict[str, Any]:
    return {
        "n_traces": len(episodic_memory),
        "check_value": compute_check_value_report(episodic_memory),
        "declining_strategies": flag_declining_strategies(procedural_memory),
        "weak_areas": weak_areas(episodic_memory, error_memory, knowledge_graph),
    }
