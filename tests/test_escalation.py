import tempfile
from pathlib import Path

import pytest

from physics_agent.self_correction.escalation import detect_escalations
from physics_agent.trace import EpisodicMemory, Trace


@pytest.fixture
def episodic_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "episodic.jsonl"


def _write(memory, domain_tags, resolution_status):
    t = Trace.new("x")
    t.domain_tags = domain_tags
    t.resolution_status = resolution_status
    memory.write(t)


def test_no_escalations_on_empty_memory(episodic_path):
    memory = EpisodicMemory(episodic_path)
    assert detect_escalations(memory) == []


def test_escalated_for_human_review_traces_are_grouped_by_domain(episodic_path):
    memory = EpisodicMemory(episodic_path)
    _write(memory, ["quantum-mechanics"], "escalated_for_human_review")
    _write(memory, ["quantum-mechanics"], "escalated_for_human_review")
    _write(memory, ["optics"], "escalated_for_human_review")
    _write(memory, ["quantum-mechanics"], "resolved_after_revision")  # should be ignored

    escalations = detect_escalations(memory, min_recurring_unresolved=999)  # disable the other signal
    sources = {e["source"] for e in escalations}
    assert sources == {"escalated_for_human_review"}

    qm_entry = next(e for e in escalations if e["domain_tags"] == ["quantum-mechanics"])
    assert qm_entry["weight"] == 2
    optics_entry = next(e for e in escalations if e["domain_tags"] == ["optics"])
    assert optics_entry["weight"] == 1


def test_recurring_unresolved_below_threshold_is_not_flagged(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(4):
        _write(memory, ["mechanics"], "unresolved_max_revisions")

    escalations = detect_escalations(memory, min_recurring_unresolved=5)
    assert escalations == []


def test_recurring_unresolved_at_threshold_is_flagged(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(5):
        _write(memory, ["mechanics"], "unresolved_max_revisions")

    escalations = detect_escalations(memory, min_recurring_unresolved=5)
    assert len(escalations) == 1
    entry = escalations[0]
    assert entry["source"] == "recurring_unresolved"
    assert entry["domain_tags"] == ["mechanics"]
    assert entry["weight"] == 5


def test_a_single_unresolved_trace_is_not_escalated_by_default(episodic_path):
    memory = EpisodicMemory(episodic_path)
    _write(memory, ["mechanics"], "unresolved_max_revisions")

    escalations = detect_escalations(memory)
    assert escalations == []


def test_escalations_sorted_by_weight_descending(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(5):
        _write(memory, ["mechanics"], "unresolved_max_revisions")
    for _ in range(8):
        _write(memory, ["optics"], "unresolved_max_revisions")

    escalations = detect_escalations(memory, min_recurring_unresolved=5)
    assert [e["domain_tags"] for e in escalations] == [["optics"], ["mechanics"]]


def test_multi_domain_trace_counted_toward_each_of_its_domains(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(5):
        _write(memory, ["mechanics", "energy"], "unresolved_max_revisions")

    escalations = detect_escalations(memory, min_recurring_unresolved=5)
    domains_flagged = {tuple(e["domain_tags"]) for e in escalations}
    assert domains_flagged == {("mechanics",), ("energy",)}
