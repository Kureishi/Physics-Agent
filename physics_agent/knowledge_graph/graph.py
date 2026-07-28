"""
Knowledge Graph (Stage 6).

Nodes are physics facts -- reuses SemanticStore's existing entries (same
ids, same confidence/provenance fields Stage 5 already updates via
record_outcome) rather than duplicating that data. This module adds the
*relational* structure on top: typed edges between facts, plus the queries
the design doc calls for:

  - check_validity: "is this formula valid under these assumptions" as a
    graph query, instead of re-deriving validity conditions from free-text
    "conditions" strings every time.
  - find_low_confidence_clusters: connected groups of shaky facts, for a
    later curriculum stage to target together rather than one at a time.
  - contradictions: facts explicitly flagged as disagreeing, surfaced
    rather than silently resolved.

Edge relations (fixed vocabulary, per the design doc):
  - derives_from:         target is a more general/foundational fact this
                          one is derived from.
  - special_case_of:       this fact is a limiting/simplified case of
                          target, valid under `condition`.
  - requires_assumption:    this fact only holds given the assumption named
                          in `condition` (target is None -- the assumption
                          itself isn't a graph node, just a label).
  - contradicts:           source and target assert incompatible things.
                          Expected to be rare; existence of one is itself
                          worth surfacing, not silently resolving.

Schema per edge:
    {id, source, target, relation, condition, confidence, provenance, last_validated}
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from ..retrieval import SemanticStore

VALID_RELATIONS = {"derives_from", "special_case_of", "requires_assumption", "contradicts"}

# Assumption tags that are detectably violated by a problem's domain_tags
# alone. This is deliberately narrow: most physics assumptions (e.g.
# "ideal_gas_approximation", "no_non_conservative_forces") can't be
# confirmed or denied from a domain tag -- they depend on the specific
# problem's setup, which this system doesn't parse into structured form.
# "non_relativistic" is the one assumption with a genuinely reliable
# tag-level signal, since a problem explicitly tagged "special-relativity"
# is a strong, direct contradiction of it.
ASSUMPTION_DOMAIN_CONFLICTS: Dict[str, Set[str]] = {
    "non_relativistic": {"special-relativity"},
}


class KnowledgeGraph:
    def __init__(self, edges_path: Union[str, Path], semantic_store: SemanticStore):
        self.semantic_store = semantic_store
        self.edges_path = Path(edges_path)
        self.edges_path.parent.mkdir(parents=True, exist_ok=True)
        if self.edges_path.exists() and self.edges_path.stat().st_size > 0:
            with self.edges_path.open("r", encoding="utf-8") as f:
                self.edges: List[Dict[str, Any]] = json.load(f)
        else:
            self.edges = []
            self._persist()

    # -- node access (delegated to SemanticStore -- single source of truth) --

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        for entry in self.semantic_store.entries:
            if entry["id"] == node_id:
                return entry
        return None

    def all_node_ids(self) -> List[str]:
        return [e["id"] for e in self.semantic_store.entries]

    # -- edge access ----------------------------------------------------------

    def add_edge(
        self,
        source: str,
        relation: str,
        condition: str,
        target: Optional[str] = None,
        confidence: float = 0.9,
        provenance: str = "manual",
    ) -> Dict[str, Any]:
        if relation not in VALID_RELATIONS:
            raise ValueError(f"Unknown relation {relation!r}; must be one of {VALID_RELATIONS}")
        edge = {
            "id": f"edge-{len(self.edges):04d}",
            "source": source,
            "target": target,
            "relation": relation,
            "condition": condition,
            "confidence": confidence,
            "provenance": provenance,
            "last_validated": time.time(),
        }
        self.edges.append(edge)
        self._persist()
        return edge

    def edges_from(self, node_id: str, relation: Optional[str] = None) -> List[Dict[str, Any]]:
        return [
            e for e in self.edges if e["source"] == node_id and (relation is None or e["relation"] == relation)
        ]

    def neighbors(self, node_id: str) -> List[str]:
        """All node ids connected to `node_id` by any edge, in either direction."""
        result: Set[str] = set()
        for e in self.edges:
            if e["source"] == node_id and e["target"]:
                result.add(e["target"])
            if e["target"] == node_id:
                result.add(e["source"])
        return sorted(result)

    # -- queries the design doc calls for --------------------------------------

    def check_validity(self, node_id: str, domain_tags: List[str]) -> Dict[str, Any]:
        """
        "Is this formula valid under these assumptions," as a graph query:
        checks every requires_assumption edge from `node_id` against
        ASSUMPTION_DOMAIN_CONFLICTS and reports which (if any) are violated
        by the problem's domain_tags.

        This deliberately only catches the narrow set of assumptions
        detectable from domain tags alone (see
        ASSUMPTION_DOMAIN_CONFLICTS's docstring). A clean result here is
        NOT proof the formula is being used correctly -- it's evidence
        against one specific, checkable way it could be wrong.
        """
        violated = []
        for edge in self.edges_from(node_id, relation="requires_assumption"):
            assumption = edge["condition"]
            conflicting_tags = ASSUMPTION_DOMAIN_CONFLICTS.get(assumption, set())
            hit = conflicting_tags & set(domain_tags)
            if hit:
                violated.append({"assumption": assumption, "conflicting_domain_tags": sorted(hit)})
        return {"valid": len(violated) == 0, "violated_assumptions": violated}

    def contradictions(self) -> List[Dict[str, Any]]:
        """
        All contradicts edges. These should be rare -- existing at all is
        itself worth surfacing (to a person, or a later meta-learning
        stage) rather than silently picking a winner between two sources.
        """
        return [e for e in self.edges if e["relation"] == "contradicts"]

    def find_low_confidence_clusters(self, threshold: float = 0.8) -> List[List[str]]:
        """
        Connected components (via any edge) among nodes whose *current*
        confidence -- read live from semantic_store, since that's what
        Stage 5 keeps updated -- is below `threshold`. This is the query a
        later autonomous-curriculum stage would run to find *related* weak
        spots to target together (e.g. a formula and the assumption-check
        that keeps failing on it), rather than one fact at a time.
        """
        low_conf_ids = {
            e["id"] for e in self.semantic_store.entries if e.get("confidence", 1.0) < threshold
        }
        visited: Set[str] = set()
        clusters: List[List[str]] = []

        for node_id in low_conf_ids:
            if node_id in visited:
                continue
            # BFS restricted to low-confidence nodes only -- a cluster is a
            # connected group of *weak* facts, not just any connected group.
            cluster = []
            queue = [node_id]
            while queue:
                current = queue.pop()
                if current in visited:
                    continue
                visited.add(current)
                cluster.append(current)
                for neighbor in self.neighbors(current):
                    if neighbor in low_conf_ids and neighbor not in visited:
                        queue.append(neighbor)
            clusters.append(sorted(cluster))

        return clusters

    def _persist(self) -> None:
        with self.edges_path.open("w", encoding="utf-8") as f:
            json.dump(self.edges, f, indent=2)
