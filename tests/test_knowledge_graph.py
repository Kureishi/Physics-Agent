import json
import tempfile
from pathlib import Path

import pytest

from physics_agent.knowledge_graph.graph import KnowledgeGraph
from physics_agent.retrieval import SemanticStore

SAMPLE_ENTRIES = [
    {
        "id": "eng-001",
        "statement": "KE = 0.5*m*v^2",
        "conditions": "Non-relativistic",
        "confidence": 0.99,
        "provenance": "seed",
        "tags": ["energy"],
        "last_validated": 0,
    },
    {
        "id": "rel-001",
        "statement": "Relativistic KE",
        "conditions": "Special relativity",
        "confidence": 0.97,
        "provenance": "seed",
        "tags": ["special-relativity", "energy"],
        "last_validated": 0,
    },
    {
        "id": "grav-001",
        "statement": "F = G*m1*m2/r^2",
        "conditions": "Point masses",
        "confidence": 0.5,  # deliberately low, for cluster tests
        "provenance": "seed",
        "tags": ["gravitation"],
        "last_validated": 0,
    },
    {
        "id": "orbit-001",
        "statement": "Orbital velocity formula",
        "conditions": "Circular orbit",
        "confidence": 0.4,  # deliberately low, connected to grav-001
        "provenance": "seed",
        "tags": ["gravitation"],
        "last_validated": 0,
    },
    {
        "id": "isolated-001",
        "statement": "Some unrelated low-confidence fact",
        "conditions": "N/A",
        "confidence": 0.3,  # low confidence but no edges -- its own cluster
        "provenance": "seed",
        "tags": ["thermodynamics"],
        "last_validated": 0,
    },
]


@pytest.fixture
def graph():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        semantic_path = d / "semantic.json"
        with semantic_path.open("w") as f:
            json.dump(SAMPLE_ENTRIES, f)
        store = SemanticStore(semantic_path)
        edges_path = d / "edges.json"
        yield KnowledgeGraph(edges_path, store)


def test_creates_empty_edge_store_if_missing(graph):
    assert graph.edges == []


def test_get_node_delegates_to_semantic_store(graph):
    node = graph.get_node("eng-001")
    assert node is not None
    assert node["statement"] == "KE = 0.5*m*v^2"


def test_get_node_returns_none_for_unknown_id(graph):
    assert graph.get_node("does-not-exist") is None


def test_all_node_ids(graph):
    ids = graph.all_node_ids()
    assert "eng-001" in ids
    assert len(ids) == len(SAMPLE_ENTRIES)


def test_add_edge_rejects_unknown_relation(graph):
    with pytest.raises(ValueError):
        graph.add_edge("eng-001", relation="not_a_real_relation", condition="x")


def test_add_edge_persists_and_returns_entry(graph):
    edge = graph.add_edge(
        "rel-001", relation="special_case_of", condition="v/c -> 0", target="eng-001", confidence=0.95
    )
    assert edge["source"] == "rel-001"
    assert edge["target"] == "eng-001"
    assert len(graph.edges) == 1

    reloaded = KnowledgeGraph(graph.edges_path, graph.semantic_store)
    assert len(reloaded.edges) == 1


def test_edges_from_filters_by_relation(graph):
    graph.add_edge("eng-001", relation="requires_assumption", condition="non_relativistic")
    graph.add_edge("eng-001", relation="derives_from", condition="x", target="grav-001")

    only_assumptions = graph.edges_from("eng-001", relation="requires_assumption")
    assert len(only_assumptions) == 1
    assert only_assumptions[0]["condition"] == "non_relativistic"

    all_from = graph.edges_from("eng-001")
    assert len(all_from) == 2


def test_neighbors_finds_both_directions(graph):
    graph.add_edge("rel-001", relation="special_case_of", condition="v/c -> 0", target="eng-001")
    assert graph.neighbors("rel-001") == ["eng-001"]
    assert graph.neighbors("eng-001") == ["rel-001"]


def test_neighbors_ignores_requires_assumption_edges_with_no_target(graph):
    graph.add_edge("eng-001", relation="requires_assumption", condition="non_relativistic")
    assert graph.neighbors("eng-001") == []


# -- check_validity ---------------------------------------------------------


def test_check_validity_passes_when_no_requires_assumption_edges(graph):
    result = graph.check_validity("eng-001", domain_tags=["special-relativity"])
    assert result["valid"] is True  # no edges yet -> nothing to violate


def test_check_validity_detects_non_relativistic_conflict(graph):
    graph.add_edge("eng-001", relation="requires_assumption", condition="non_relativistic")
    result = graph.check_validity("eng-001", domain_tags=["special-relativity", "energy"])
    assert result["valid"] is False
    assert result["violated_assumptions"][0]["assumption"] == "non_relativistic"
    assert "special-relativity" in result["violated_assumptions"][0]["conflicting_domain_tags"]


def test_check_validity_passes_when_domain_tags_dont_conflict(graph):
    graph.add_edge("eng-001", relation="requires_assumption", condition="non_relativistic")
    result = graph.check_validity("eng-001", domain_tags=["dynamics", "energy"])
    assert result["valid"] is True


def test_check_validity_ignores_unmappable_assumptions(graph):
    # "ideal_gas_approximation" has no entry in ASSUMPTION_DOMAIN_CONFLICTS
    # -- can't be verified from domain tags alone, so it should never
    # itself cause a failure no matter what domain tags are given.
    graph.add_edge("eng-001", relation="requires_assumption", condition="ideal_gas_approximation")
    result = graph.check_validity("eng-001", domain_tags=["thermodynamics"])
    assert result["valid"] is True


# -- contradictions -----------------------------------------------------------


def test_contradictions_empty_by_default(graph):
    assert graph.contradictions() == []


def test_contradictions_returns_only_contradicts_edges(graph):
    graph.add_edge("eng-001", relation="requires_assumption", condition="non_relativistic")
    graph.add_edge("eng-001", relation="contradicts", condition="disagrees under X", target="rel-001")
    contradictions = graph.contradictions()
    assert len(contradictions) == 1
    assert contradictions[0]["relation"] == "contradicts"


# -- find_low_confidence_clusters ---------------------------------------------


def test_low_confidence_clusters_groups_connected_weak_nodes(graph):
    # grav-001 (0.5) and orbit-001 (0.4) are connected and both below 0.8
    graph.add_edge("orbit-001", relation="derives_from", condition="x", target="grav-001")
    clusters = graph.find_low_confidence_clusters(threshold=0.8)

    grav_cluster = next(c for c in clusters if "grav-001" in c)
    assert set(grav_cluster) == {"grav-001", "orbit-001"}


def test_low_confidence_clusters_separates_unconnected_nodes(graph):
    graph.add_edge("orbit-001", relation="derives_from", condition="x", target="grav-001")
    clusters = graph.find_low_confidence_clusters(threshold=0.8)

    # isolated-001 (0.3, no edges) should be its own singleton cluster,
    # separate from the grav-001/orbit-001 pair
    isolated_cluster = next(c for c in clusters if "isolated-001" in c)
    assert isolated_cluster == ["isolated-001"]
    assert len(clusters) == 2  # {grav-001, orbit-001} and {isolated-001}


def test_low_confidence_clusters_excludes_high_confidence_nodes(graph):
    clusters = graph.find_low_confidence_clusters(threshold=0.8)
    all_clustered = {n for cluster in clusters for n in cluster}
    assert "eng-001" not in all_clustered  # confidence 0.99, above threshold
    assert "rel-001" not in all_clustered  # confidence 0.97, above threshold


def test_low_confidence_clusters_empty_when_threshold_very_low(graph):
    clusters = graph.find_low_confidence_clusters(threshold=0.1)
    assert clusters == []
