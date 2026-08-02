import json

import pytest

from physics_agent.config import Config
from physics_agent.generate_problem_set_cli import generate_for_domains, load_existing
from physics_agent.planner import DOMAIN_TAXONOMY


@pytest.fixture
def config(tmp_path):
    return Config(
        semantic_store_path=str(tmp_path / "semantic.json"),
        knowledge_graph_path=str(tmp_path / "edges.json"),
        episodic_memory_path=str(tmp_path / "episodic.jsonl"),
        procedural_memory_path=str(tmp_path / "procedural.json"),
        error_memory_path=str(tmp_path / "error_memory.json"),
        curriculum_log_path=str(tmp_path / "curriculum_log.jsonl"),
    )


def test_generate_for_domains_produces_expected_count(config):
    problems = generate_for_domains(["energy", "dynamics"], n_per_domain=2, dry_run=True, config=config)
    assert len(problems) == 4  # 2 domains * 2 each


def test_generate_for_domains_tags_domain_hint_correctly(config):
    problems = generate_for_domains(["energy"], n_per_domain=3, dry_run=True, config=config)
    assert all(p["domain_hint"] == "energy" for p in problems)


def test_generate_for_domains_produces_unique_ids(config):
    problems = generate_for_domains(["energy", "dynamics"], n_per_domain=3, dry_run=True, config=config)
    ids = [p["id"] for p in problems]
    assert len(ids) == len(set(ids))


def test_generate_for_domains_each_has_nonempty_problem_text(config):
    problems = generate_for_domains(["gravitation"], n_per_domain=2, dry_run=True, config=config)
    assert all(p["problem_text"] for p in problems)


def test_generate_for_domains_passes_growing_avoid_list(config, monkeypatch):
    from physics_agent.curriculum.problem_generator import ProblemGenerator

    seen_avoid_lengths = []
    original_generate = ProblemGenerator.generate

    def spy_generate(self, signal, avoid=None):
        seen_avoid_lengths.append(len(avoid) if avoid else 0)
        return original_generate(self, signal, avoid=avoid)

    monkeypatch.setattr(ProblemGenerator, "generate", spy_generate)

    generate_for_domains(["energy"], n_per_domain=3, dry_run=True, config=config)

    # first call has nothing to avoid yet; each subsequent call should see
    # one more prior problem to avoid duplicating
    assert seen_avoid_lengths == [0, 1, 2]


def test_generate_for_domains_skips_failed_generation_without_crashing(config, monkeypatch):
    from physics_agent.curriculum.problem_generator import ProblemGenerator

    call_count = {"n": 0}
    original_generate = ProblemGenerator.generate

    def flaky_generate(self, signal, avoid=None):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise ValueError("simulated failure")
        return original_generate(self, signal, avoid=avoid)

    monkeypatch.setattr(ProblemGenerator, "generate", flaky_generate)

    problems = generate_for_domains(["energy"], n_per_domain=3, dry_run=True, config=config)
    assert len(problems) == 2  # one of the three failed and was skipped


def test_generate_for_domains_survives_a_non_valueerror_exception(config, monkeypatch):
    # Defense-in-depth: even if something other than ProblemGenerator's own
    # (always-ValueError) failure mode somehow leaked through, the batch
    # should still survive it rather than crashing entirely -- this is what
    # actually happened in practice with an uncaught openai.BadRequestError
    # before generate()'s retry loop was hardened to catch it internally.
    from physics_agent.curriculum.problem_generator import ProblemGenerator

    call_count = {"n": 0}
    original_generate = ProblemGenerator.generate

    def flaky_generate(self, signal, avoid=None):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated raw API-level failure")
        return original_generate(self, signal, avoid=avoid)

    monkeypatch.setattr(ProblemGenerator, "generate", flaky_generate)

    problems = generate_for_domains(["energy"], n_per_domain=3, dry_run=True, config=config)
    assert len(problems) == 2


def test_load_existing_returns_empty_list_for_missing_file(tmp_path):
    result = load_existing(str(tmp_path / "does_not_exist.json"))
    assert result == []


def test_load_existing_reads_real_file(tmp_path):
    path = tmp_path / "existing.json"
    data = [{"id": "a", "domain_hint": "energy", "problem_text": "x"}]
    with path.open("w") as f:
        json.dump(data, f)
    assert load_existing(str(path)) == data


def test_domain_taxonomy_has_no_duplicates_or_empty_entries():
    # Deliberately does NOT hardcode a count -- planner.DOMAIN_TAXONOMY is
    # expected to grow over time as new domains get added, and a test
    # asserting an exact number would just be one more place someone has
    # to remember to update alongside it.
    assert len(DOMAIN_TAXONOMY) == len(set(DOMAIN_TAXONOMY))  # no duplicates
    assert all(isinstance(tag, str) and tag for tag in DOMAIN_TAXONOMY)  # no empty/non-string entries
