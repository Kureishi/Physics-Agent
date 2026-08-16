import json

import pytest

from physics_agent.canary.problems import CanaryProblem
from physics_agent.canary.runner import (
    CanaryLog,
    CanaryRunner,
    _classify,
    latest_result_per_canary,
    summarize,
)
from physics_agent.config import Config


@pytest.fixture
def config(tmp_path):
    semantic_path = tmp_path / "semantic.json"
    with semantic_path.open("w") as f:
        json.dump([], f)
    return Config(
        semantic_store_path=str(semantic_path),
        knowledge_graph_path=str(tmp_path / "edges.json"),
        episodic_memory_path=str(tmp_path / "episodic.jsonl"),
        procedural_memory_path=str(tmp_path / "procedural.json"),
        error_memory_path=str(tmp_path / "error_memory.json"),
        canary_log_path=str(tmp_path / "canary_log.jsonl"),
    )


# Under dry_run (MockLLMClient), tool selection is fixed regardless of
# problem text: it always solves Eq(m*g*h, 0.5*m*v**2) with m=2, g=9.8,
# h=5 -> v = sqrt(2*9.8*5) = 9.899494936611665, and every LLM-based check
# defaults to "passed". This makes the mock pipeline deterministic enough
# to test grading/verdict logic without needing a real LM Studio server.
_MOCK_SOLVED_VALUE = 9.899494936611665


def _correct_canary() -> CanaryProblem:
    return CanaryProblem(
        id="canary-correct",
        domain_hint="dynamics",
        problem_text="irrelevant under mock tool selection",
        expected_value=_MOCK_SOLVED_VALUE,
        relative_tolerance=0.02,
        units="m/s",
        verified_by="test fixture",
    )


def _incorrect_canary() -> CanaryProblem:
    return CanaryProblem(
        id="canary-incorrect",
        domain_hint="dynamics",
        problem_text="irrelevant under mock tool selection",
        expected_value=500.0,
        relative_tolerance=0.02,
        units="m/s",
        verified_by="test fixture",
    )


def test_classify_correct_and_passed():
    assert _classify(answer_correct=True, n_candidates=1, checks_failed=[]) == "correct_and_passed"


def test_classify_correct_but_flagged():
    assert (
        _classify(answer_correct=True, n_candidates=1, checks_failed=["math"])
        == "correct_but_flagged"
    )


def test_classify_incorrect_but_passed():
    assert (
        _classify(answer_correct=False, n_candidates=1, checks_failed=[])
        == "incorrect_but_passed"
    )


def test_classify_incorrect_and_flagged():
    assert (
        _classify(answer_correct=False, n_candidates=1, checks_failed=["physics"])
        == "incorrect_and_flagged"
    )


def test_classify_unmeasurable_overrides_answer_correct():
    assert _classify(answer_correct=False, n_candidates=0, checks_failed=["math"]) == "unmeasurable"


def test_run_all_raises_on_empty_problem_set(config):
    runner = CanaryRunner(config, dry_run=True)
    with pytest.raises(ValueError):
        runner.run_all(problems=[])


def test_run_all_grades_correct_canary_as_correct_and_passed(config):
    runner = CanaryRunner(config, dry_run=True)
    results = runner.run_all(problems=[_correct_canary()])

    assert len(results) == 1
    r = results[0]
    assert r.canary_id == "canary-correct"
    assert r.answer_correct is True
    assert r.verdict == "correct_and_passed"
    assert r.checks_failed == []


def test_run_all_grades_incorrect_canary_as_incorrect_but_passed(config):
    runner = CanaryRunner(config, dry_run=True)
    results = runner.run_all(problems=[_incorrect_canary()])

    assert len(results) == 1
    r = results[0]
    assert r.canary_id == "canary-incorrect"
    assert r.answer_correct is False
    # Mock LLM checks all default to "passed" -- so a wrong answer here
    # should surface as the dangerous "checks passed anyway" verdict,
    # exactly the case this feature is built to catch.
    assert r.verdict == "incorrect_but_passed"


def test_run_all_persists_to_canary_log(config):
    runner = CanaryRunner(config, dry_run=True)
    runner.run_all(problems=[_correct_canary(), _incorrect_canary()])

    log = CanaryLog(config.canary_log_path)
    entries = log.read_all()
    assert len(entries) == 2
    assert {e.canary_id for e in entries} == {"canary-correct", "canary-incorrect"}


def test_summarize_counts_verdicts(config):
    runner = CanaryRunner(config, dry_run=True)
    results = runner.run_all(problems=[_correct_canary(), _incorrect_canary()])

    counts = summarize(results)
    assert counts["correct_and_passed"] == 1
    assert counts["incorrect_but_passed"] == 1
    assert counts["correct_but_flagged"] == 0
    assert counts["incorrect_and_flagged"] == 0
    assert counts["unmeasurable"] == 0


def test_latest_result_per_canary_picks_most_recent(config):
    runner = CanaryRunner(config, dry_run=True)
    runner.run_all(problems=[_correct_canary()])
    runner.run_all(problems=[_incorrect_canary()])  # same id reused below

    log = CanaryLog(config.canary_log_path)
    entries = log.read_all()
    # Force a second, later run of the same canary id with a different
    # outcome and confirm latest_result_per_canary picks it up.
    entries[0].canary_id = "shared-id"
    entries[0].timestamp = 1.0
    entries[0].verdict = "correct_and_passed"
    entries[1].canary_id = "shared-id"
    entries[1].timestamp = 2.0
    entries[1].verdict = "incorrect_but_passed"

    latest = latest_result_per_canary(entries)
    assert latest["shared-id"].verdict == "incorrect_but_passed"
