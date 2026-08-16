import json

import pytest

from physics_agent.scheduler.queue import ProblemQueue


@pytest.fixture
def queue_path(tmp_path):
    return tmp_path / "queue.json"


def test_creates_empty_queue_if_missing(queue_path):
    q = ProblemQueue(queue_path)
    assert len(q) == 0
    assert queue_path.exists()


def test_push_and_pop_fifo_order(queue_path):
    q = ProblemQueue(queue_path)
    q.push({"id": "a", "problem_text": "first"})
    q.push({"id": "b", "problem_text": "second"})

    first = q.pop_next()
    assert first["id"] == "a"
    assert len(q) == 1

    second = q.pop_next()
    assert second["id"] == "b"
    assert len(q) == 0


def test_pop_next_on_empty_queue_returns_none(queue_path):
    q = ProblemQueue(queue_path)
    assert q.pop_next() is None


def test_extend_adds_multiple(queue_path):
    q = ProblemQueue(queue_path)
    q.extend([{"id": "a"}, {"id": "b"}, {"id": "c"}])
    assert len(q) == 3
    assert [p["id"] for p in q.peek_all()] == ["a", "b", "c"]


def test_pop_persists_across_instances(queue_path):
    q1 = ProblemQueue(queue_path)
    q1.push({"id": "a", "problem_text": "x"})

    q2 = ProblemQueue(queue_path)  # simulates a fresh process
    assert len(q2) == 1
    popped = q2.pop_next()
    assert popped["id"] == "a"

    q3 = ProblemQueue(queue_path)
    assert len(q3) == 0


def test_reads_externally_supplied_problem_set_shaped_file(queue_path):
    # Same shape as data/problem_sets/*.json -- confirms the queue can be
    # pointed directly at a hand-written or generate_problem_set_cli
    # output file with no adapter.
    with queue_path.open("w") as f:
        json.dump(
            [
                {"id": "ext-1", "domain_hint": "energy", "problem_text": "problem one"},
                {"id": "ext-2", "domain_hint": "dynamics", "problem_text": "problem two"},
            ],
            f,
        )

    q = ProblemQueue(queue_path)
    assert len(q) == 2
    first = q.pop_next()
    assert first == {"id": "ext-1", "domain_hint": "energy", "problem_text": "problem one"}
