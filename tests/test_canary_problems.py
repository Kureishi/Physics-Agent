import json

import pytest

from physics_agent.canary.problems import CanaryProblem, load_canary_problems


def test_loads_real_data_file():
    problems = load_canary_problems("data/canary_problems.json")
    assert len(problems) >= 5
    assert all(isinstance(p, CanaryProblem) for p in problems)


def test_real_data_file_has_unique_ids():
    problems = load_canary_problems("data/canary_problems.json")
    ids = [p.id for p in problems]
    assert len(ids) == len(set(ids))


def test_real_data_file_entries_have_sane_fields():
    problems = load_canary_problems("data/canary_problems.json")
    for p in problems:
        assert p.problem_text
        assert p.domain_hint
        assert isinstance(p.expected_value, float)
        assert 0 < p.relative_tolerance < 1


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_canary_problems(tmp_path / "does_not_exist.json")


def test_load_defaults_missing_optional_fields(tmp_path):
    path = tmp_path / "canaries.json"
    with path.open("w") as f:
        json.dump(
            [
                {
                    "id": "canary-x",
                    "domain_hint": "kinematics",
                    "problem_text": "trivial",
                    "expected_value": 1.0,
                }
            ],
            f,
        )
    problems = load_canary_problems(path)
    assert len(problems) == 1
    assert problems[0].relative_tolerance == 0.02
    assert problems[0].units == ""
    assert problems[0].verified_by == ""
