"""
Tool Registry (Stage 2) -- the "Physics Tool Selection" box in the
architecture diagram.

Holds the tool implementations, and maps physics domain tags (from Stage
1's Task Planner) to which tools are typically relevant. This restricts
what the LLM tool-selection call is offered, so it isn't asked to choose
among tools irrelevant to the problem (e.g. it's not offered
literature_search for a basic kinematics problem, which keeps that call
faster and reduces hallucinated/irrelevant tool use).
"""
from __future__ import annotations

from typing import Dict, List

from .literature import LiteratureSearchTool
from .simulation import SimulationTool
from .symbolic_math import SymbolicMathTool

DOMAIN_TOOL_HINTS: Dict[str, List[str]] = {
    "kinematics": ["symbolic_math", "simulation"],
    "dynamics": ["symbolic_math", "simulation"],
    "energy": ["symbolic_math", "simulation"],
    "momentum": ["symbolic_math", "simulation"],
    "rotational-dynamics": ["symbolic_math", "simulation"],
    "gravitation": ["symbolic_math", "simulation", "literature_search"],
    "oscillations-waves": ["symbolic_math", "simulation"],
    "thermodynamics": ["symbolic_math", "literature_search"],
    "electromagnetism": ["symbolic_math", "simulation", "literature_search"],
    "optics": ["symbolic_math", "literature_search"],
    "fluid-mechanics": ["symbolic_math", "simulation", "literature_search"],
    "special-relativity": ["symbolic_math", "literature_search"],
    "quantum-mechanics": ["symbolic_math", "literature_search"],
    "statistical-mechanics": ["symbolic_math", "literature_search"],
}


class ToolRegistry:
    def __init__(self):
        self._tools = {
            "symbolic_math": SymbolicMathTool(),
            "simulation": SimulationTool(),
            "literature_search": LiteratureSearchTool(),
        }

    def get(self, name: str):
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name!r}. Available tools: {list(self._tools)}")
        return self._tools[name]

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def relevant_tools(self, domain_tags: List[str]) -> List[str]:
        """
        Union of tool hints across all of a problem's domain tags. Falls
        back to offering all tools if no tags matched anything (safer
        default than offering none, e.g. for an unclassified problem).
        """
        relevant = set()
        for tag in domain_tags:
            relevant.update(DOMAIN_TOOL_HINTS.get(tag, []))
        return sorted(relevant) if relevant else self.names()
