import json

import pytest

from physics_agent.config import Config
from physics_agent.problem_set_cli import load_problem_set, run_problem_set


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


def test_load_problem_set_reads_real_file():
    problems = load_problem_set("data/problem_sets/intro_physics_set.json")
    assert len(problems) >= 20
    assert all("id" in p and "problem_text" in p for p in problems)
    # ids should be unique
    assert len(problems) == len(set(p["id"] for p in problems))


def test_run_problem_set_dry_run_solves_all(config):
    problems = [
        {"id": "p1", "domain_hint": "energy", "problem_text": "A block slides down a frictionless incline."},
        {"id": "p2", "domain_hint": "dynamics", "problem_text": "A box is pushed across a floor."},
    ]
    results = run_problem_set(problems, dry_run=True, config=config)

    assert len(results) == 2
    assert all("trace" in r for r in results)
    assert all(r["trace"].resolution_status is not None for r in results)


def test_run_problem_set_respects_limit(config):
    problems = [
        {"id": f"p{i}", "domain_hint": "energy", "problem_text": f"problem {i}"} for i in range(5)
    ]
    results = run_problem_set(problems, dry_run=True, config=config, limit=2)
    assert len(results) == 2


def test_run_problem_set_survives_a_crashing_problem(config, monkeypatch):
    from physics_agent import cli as cli_module

    call_count = {"n": 0}
    original_run = cli_module.run

    def flaky_run(problem_text, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated crash")
        return original_run(problem_text, **kwargs)

    monkeypatch.setattr(cli_module, "run", flaky_run)

    problems = [
        {"id": "p1", "domain_hint": "energy", "problem_text": "crashes"},
        {"id": "p2", "domain_hint": "energy", "problem_text": "solves fine"},
    ]
    results = run_problem_set(problems, dry_run=True, config=config)

    assert len(results) == 2
    assert "error" in results[0]
    assert "trace" in results[1]
