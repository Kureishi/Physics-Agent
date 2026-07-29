import tempfile
from pathlib import Path

import pytest

from physics_agent.meta_learning.verification_depth import (
    VerificationDepthPolicy,
    MIN_TRACES_BEFORE_ACTING,
    OVERCONFIDENCE_GAP_THRESHOLD,
    MAX_THRESHOLD,
)
from physics_agent.trace import EpisodicMemory, Trace


@pytest.fixture
def episodic_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "episodic.jsonl"


def _make_trace(domain_tags, confidence, resolution_status):
    trace = Trace.new("test problem")
    trace.domain_tags = domain_tags
    trace.final_confidence = confidence
    trace.resolution_status = resolution_status
    return trace


def test_no_data_returns_none_gap(episodic_path):
    memory = EpisodicMemory(episodic_path)
    policy = VerificationDepthPolicy(memory)
    assert policy.overconfidence_gap("energy") is None


def test_gap_requires_minimum_traces(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(MIN_TRACES_BEFORE_ACTING - 1):
        memory.write(_make_trace(["energy"], 0.9, "unresolved_max_revisions"))
    policy = VerificationDepthPolicy(memory)
    assert policy.overconfidence_gap("energy") is None


def test_overconfident_domain_shows_positive_gap(episodic_path):
    memory = EpisodicMemory(episodic_path)
    # high confidence, but frequently unresolved -- classic overconfidence
    for _ in range(MIN_TRACES_BEFORE_ACTING):
        memory.write(_make_trace(["quantum-mechanics"], 0.9, "unresolved_max_revisions"))
    policy = VerificationDepthPolicy(memory)
    gap = policy.overconfidence_gap("quantum-mechanics")
    assert gap is not None
    assert gap > 0.5  # avg_confidence=0.9, trust_warranted=0 -> gap=0.9


def test_well_calibrated_domain_shows_low_or_zero_gap(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(MIN_TRACES_BEFORE_ACTING):
        memory.write(_make_trace(["dynamics"], 0.85, "passed_initial"))
    policy = VerificationDepthPolicy(memory)
    gap = policy.overconfidence_gap("dynamics")
    assert gap is not None
    assert gap < OVERCONFIDENCE_GAP_THRESHOLD


def test_recommended_threshold_stays_default_with_no_data(episodic_path):
    memory = EpisodicMemory(episodic_path)
    policy = VerificationDepthPolicy(memory)
    assert policy.recommended_confidence_threshold(["energy"], default_threshold=0.6) == 0.6


def test_recommended_threshold_stays_default_when_well_calibrated(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(MIN_TRACES_BEFORE_ACTING):
        memory.write(_make_trace(["dynamics"], 0.85, "passed_initial"))
    policy = VerificationDepthPolicy(memory)
    assert policy.recommended_confidence_threshold(["dynamics"], default_threshold=0.6) == 0.6


def test_recommended_threshold_rises_when_overconfident(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(MIN_TRACES_BEFORE_ACTING):
        memory.write(_make_trace(["quantum-mechanics"], 0.9, "unresolved_max_revisions"))
    policy = VerificationDepthPolicy(memory)
    threshold = policy.recommended_confidence_threshold(["quantum-mechanics"], default_threshold=0.6)
    assert threshold > 0.6


def test_recommended_threshold_never_exceeds_max(episodic_path):
    memory = EpisodicMemory(episodic_path)
    # extreme overconfidence: max possible confidence, always unresolved
    for _ in range(MIN_TRACES_BEFORE_ACTING):
        memory.write(_make_trace(["quantum-mechanics"], 1.0, "unresolved_max_revisions"))
    policy = VerificationDepthPolicy(memory)
    threshold = policy.recommended_confidence_threshold(["quantum-mechanics"], default_threshold=0.6)
    assert threshold <= MAX_THRESHOLD


def test_recommended_threshold_never_drops_below_default(episodic_path):
    # Even a domain that looks "underconfident" (low confidence, but always
    # resolves fine) should never get a LOWER threshold than the default --
    # this policy only ever raises the bar, never lowers it.
    memory = EpisodicMemory(episodic_path)
    for _ in range(MIN_TRACES_BEFORE_ACTING):
        memory.write(_make_trace(["dynamics"], 0.2, "passed_initial"))
    policy = VerificationDepthPolicy(memory)
    threshold = policy.recommended_confidence_threshold(["dynamics"], default_threshold=0.6)
    assert threshold == 0.6


def test_uses_max_gap_across_multiple_domain_tags(episodic_path):
    memory = EpisodicMemory(episodic_path)
    for _ in range(MIN_TRACES_BEFORE_ACTING):
        memory.write(_make_trace(["dynamics"], 0.85, "passed_initial"))  # well-calibrated
        memory.write(_make_trace(["quantum-mechanics"], 0.9, "unresolved_max_revisions"))  # overconfident
    policy = VerificationDepthPolicy(memory)
    # a problem tagged with BOTH should inherit the higher (more cautious) threshold
    threshold = policy.recommended_confidence_threshold(["dynamics", "quantum-mechanics"], default_threshold=0.6)
    assert threshold > 0.6
