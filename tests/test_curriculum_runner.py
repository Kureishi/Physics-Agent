import json

import pytest

from physics_agent.config import Config
from physics_agent.curriculum.curriculum_runner import CurriculumLog, CurriculumRunner
from physics_agent.memory.error_memory import ErrorMemory
from physics_agent.trace import EpisodicMemory


@pytest.fixture
def config(tmp_path):
    semantic_path = tmp_path / "semantic.json"
    with semantic_path.open("w") as f:
        json.dump(
            [
                {
                    "id": "eng-001",
                    "statement": "KE = 0.5*m*v^2",
                    "conditions": "Non-relativistic",
                    "confidence": 0.99,
                    "provenance": "seed",
                    "tags": ["energy"],
                    "last_validated": 0,
                }
            ],
            f,
        )
    return Config(
        semantic_store_path=str(semantic_path),
        knowledge_graph_path=str(tmp_path / "edges.json"),
        episodic_memory_path=str(tmp_path / "episodic.jsonl"),
        procedural_memory_path=str(tmp_path / "procedural.json"),
        error_memory_path=str(tmp_path / "error_memory.json"),
        curriculum_log_path=str(tmp_path / "curriculum_log.jsonl"),
    )


def test_run_round_with_no_weak_areas_returns_empty(config):
    runner = CurriculumRunner(config, dry_run=True)
    results = runner.run_round(n_problems=1)
    assert results == []


def test_run_round_generates_and_solves_targeting_error_memory_signal(config):
    error_memory = ErrorMemory(config.error_memory_path)
    error_memory.record("algebra_error", ["energy"], "bad algebra", "rederive_math", resolved=True)

    runner = CurriculumRunner(config, dry_run=True)
    results = runner.run_round(n_problems=1)

    assert len(results) == 1
    r = results[0]
    assert r.targeted_signal["source"] == "error_memory"
    assert r.generated_problem_text
    assert r.resolution_status is not None  # actually solved via the real pipeline


def test_generated_problem_is_tagged_curriculum_in_episodic_memory(config):
    error_memory = ErrorMemory(config.error_memory_path)
    error_memory.record("algebra_error", ["energy"], "bad algebra", "rederive_math", resolved=True)

    runner = CurriculumRunner(config, dry_run=True)
    runner.run_round(n_problems=1)

    episodic = EpisodicMemory(config.episodic_memory_path)
    curriculum_traces = episodic.query_by_source("curriculum")
    assert len(curriculum_traces) == 1
    assert curriculum_traces[0].curriculum_target["source"] == "error_memory"


def test_curriculum_log_persists_round_results(config):
    error_memory = ErrorMemory(config.error_memory_path)
    error_memory.record("algebra_error", ["energy"], "bad algebra", "rederive_math", resolved=True)

    runner = CurriculumRunner(config, dry_run=True)
    runner.run_round(n_problems=1)

    log = CurriculumLog(config.curriculum_log_path)
    entries = log.read_all()
    assert len(entries) == 1
    assert entries[0]["targeted_signal"]["source"] == "error_memory"
    assert entries[0]["metric_description"]


def test_error_memory_metric_measured_before_and_after(config):
    error_memory = ErrorMemory(config.error_memory_path)
    error_memory.record("algebra_error", ["energy"], "bad algebra", "rederive_math", resolved=True)

    runner = CurriculumRunner(config, dry_run=True)
    results = runner.run_round(n_problems=1)

    r = results[0]
    assert r.metric_before == 1.0  # one prior recorded occurrence
    # after: depends on whether the practice problem also hit an algebra
    # error -- either way it must be a real number, not None, since
    # error_memory frequency is always measurable once the signature exists
    assert r.metric_after is not None


def test_multiple_signals_each_get_a_result(config):
    error_memory = ErrorMemory(config.error_memory_path)
    error_memory.record("algebra_error", ["energy"], "bad algebra", "rederive_math", resolved=True)
    error_memory.record("physics_conceptual_error", ["dynamics"], "bad physics", "rederive_physics_setup", resolved=True)

    runner = CurriculumRunner(config, dry_run=True)
    results = runner.run_round(n_problems=2)

    assert len(results) == 2
    sources_targeted = {r.targeted_signal["error_type"] for r in results}
    assert sources_targeted == {"algebra_error", "physics_conceptual_error"}


def test_run_round_skips_signal_when_generation_fails(config, monkeypatch):
    error_memory = ErrorMemory(config.error_memory_path)
    error_memory.record("algebra_error", ["energy"], "bad algebra", "rederive_math", resolved=True)

    from physics_agent.curriculum.problem_generator import ProblemGenerator

    def always_fail(self, signal):
        raise ValueError("simulated generation failure")

    monkeypatch.setattr(ProblemGenerator, "generate", always_fail)

    runner = CurriculumRunner(config, dry_run=True)
    results = runner.run_round(n_problems=1)

    assert results == []  # the only candidate signal failed generation, gracefully skipped
