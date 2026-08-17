import json
from pathlib import Path

import pytest

from physics_agent.config import Config
from physics_agent.memory.error_memory import ErrorMemory
from physics_agent.scheduler.queue import ProblemQueue
from physics_agent.scheduler.scheduler import DecisionLog, Scheduler
from physics_agent.trace import EpisodicMemory, Trace, ToolCall


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
        curriculum_log_path=str(tmp_path / "curriculum_log.jsonl"),
        scheduler_queue_path=str(tmp_path / "queue.json"),
        scheduler_state_path=str(tmp_path / "scheduler_state.json"),
        scheduler_log_path=str(tmp_path / "scheduler_log.jsonl"),
        proposed_facts_registry_path=str(tmp_path / "proposed_facts.json"),
        # Defaults that keep review/practice/growth from firing
        # accidentally in tests that aren't specifically exercising them.
        scheduler_review_every_n_solves=1000,
        scheduler_curriculum_weight_threshold=1000,
        scheduler_curriculum_min_cycles_between_rounds=1000,
        scheduler_growth_min_cycles_between_rounds=1000,
    )


def _push_problem(config, problem_id="p1", text="A 2 kg block starts at rest at the top of a 5 m frictionless incline. Find its speed at the bottom."):
    queue = ProblemQueue(config.scheduler_queue_path)
    queue.push({"id": problem_id, "domain_hint": "dynamics", "problem_text": text})


def test_idle_cycle_on_empty_queue(config):
    scheduler = Scheduler(config, dry_run=True)
    decisions = scheduler.run_cycle()

    assert len(decisions) == 1
    assert decisions[0].action == "idle"
    assert scheduler.state.total_cycles == 1


def test_solve_decision_pops_queue_and_writes_episodic_trace(config):
    _push_problem(config)
    scheduler = Scheduler(config, dry_run=True)

    decisions = scheduler.run_cycle()

    solve_decisions = [d for d in decisions if d.action == "solve"]
    assert len(solve_decisions) == 1
    assert solve_decisions[0].details["problem_id"] == "p1"
    assert solve_decisions[0].details["resolution_status"] is not None

    queue = ProblemQueue(config.scheduler_queue_path)
    assert len(queue) == 0  # consumed

    episodic = EpisodicMemory(config.episodic_memory_path)
    assert len(episodic) == 1

    assert scheduler.state.total_solves == 1
    assert scheduler.state.solves_since_last_review == 1


def test_solve_decision_is_logged_to_decision_log(config):
    _push_problem(config)
    scheduler = Scheduler(config, dry_run=True)
    scheduler.run_cycle()

    log = DecisionLog(config.scheduler_log_path)
    entries = log.read_all()
    assert any(d.action == "solve" for d in entries)


def test_state_persists_across_scheduler_instances(config):
    _push_problem(config)
    scheduler1 = Scheduler(config, dry_run=True)
    scheduler1.run_cycle()

    scheduler2 = Scheduler(config, dry_run=True)
    assert scheduler2.state.total_solves == 1
    assert scheduler2.state.total_cycles == 1


def test_review_does_not_trigger_before_threshold(config):
    config.scheduler_review_every_n_solves = 3
    _push_problem(config, "p1")
    scheduler = Scheduler(config, dry_run=True)

    decisions = scheduler.run_cycle()
    assert not any(d.action == "review" for d in decisions)
    assert scheduler.state.solves_since_last_review == 1


def test_review_triggers_after_n_solves(config):
    config.scheduler_review_every_n_solves = 2
    scheduler = Scheduler(config, dry_run=True)

    _push_problem(config, "p1")
    decisions1 = scheduler.run_cycle()
    assert not any(d.action == "review" for d in decisions1)

    _push_problem(config, "p2")
    decisions2 = scheduler.run_cycle()
    review_decisions = [d for d in decisions2 if d.action == "review"]
    assert len(review_decisions) == 1
    assert review_decisions[0].details["n_traces"] == 2

    assert scheduler.state.total_reviews == 1
    assert scheduler.state.solves_since_last_review == 0


def test_review_does_not_double_count_after_firing(config):
    config.scheduler_review_every_n_solves = 1
    scheduler = Scheduler(config, dry_run=True)

    _push_problem(config, "p1")
    scheduler.run_cycle()
    assert scheduler.state.total_reviews == 1

    # Next cycle: queue is empty, no new solves -- review must not fire
    # again just because it fired once before.
    decisions = scheduler.run_cycle()
    assert not any(d.action == "review" for d in decisions)
    assert scheduler.state.total_reviews == 1


def test_practice_does_not_trigger_below_weight_threshold(config):
    config.scheduler_curriculum_weight_threshold = 5
    config.scheduler_curriculum_min_cycles_between_rounds = 1
    error_memory = ErrorMemory(config.error_memory_path)
    error_memory.record("algebra_error", ["energy"], "bad algebra", "rederive_math", resolved=True)
    # frequency == 1, below threshold of 5

    scheduler = Scheduler(config, dry_run=True)
    decisions = scheduler.run_cycle()
    assert not any(d.action == "practice" for d in decisions)


def test_practice_triggers_when_weight_threshold_cleared(config):
    config.scheduler_curriculum_weight_threshold = 3
    config.scheduler_curriculum_min_cycles_between_rounds = 1
    error_memory = ErrorMemory(config.error_memory_path)
    for _ in range(3):
        error_memory.record("algebra_error", ["energy"], "bad algebra", "rederive_math", resolved=True)
    # frequency == 3, meets threshold

    scheduler = Scheduler(config, dry_run=True)
    decisions = scheduler.run_cycle()

    practice_decisions = [d for d in decisions if d.action == "practice"]
    assert len(practice_decisions) == 1
    assert practice_decisions[0].details["targeted_signal"]["source"] == "error_memory"
    assert scheduler.state.total_curriculum_rounds == 1
    assert scheduler.state.cycles_since_last_curriculum == 0
    assert Path(config.curriculum_log_path).exists()


def test_practice_respects_cooldown_between_rounds(config):
    config.scheduler_curriculum_weight_threshold = 3
    config.scheduler_curriculum_min_cycles_between_rounds = 3
    error_memory = ErrorMemory(config.error_memory_path)
    for _ in range(5):
        error_memory.record("algebra_error", ["energy"], "bad algebra", "rederive_math", resolved=True)

    scheduler = Scheduler(config, dry_run=True)

    # cycles_since_last_curriculum starts at 0, incremented to 1 on the
    # first cycle -- below the cooldown of 3, so no practice yet.
    decisions1 = scheduler.run_cycle()
    assert not any(d.action == "practice" for d in decisions1)

    decisions2 = scheduler.run_cycle()
    assert not any(d.action == "practice" for d in decisions2)

    # Third cycle: cycles_since_last_curriculum reaches 3 -- now it fires.
    decisions3 = scheduler.run_cycle()
    assert any(d.action == "practice" for d in decisions3)


def test_practice_can_fire_on_an_otherwise_idle_cycle(config):
    # No queue activity at all -- practice should still fire purely off
    # the standing weak-area signal, per the design doc ("If weak_areas()
    # crosses a weight threshold, automatically triggers a curriculum
    # round" -- not gated on a solve happening this cycle).
    config.scheduler_curriculum_weight_threshold = 3
    config.scheduler_curriculum_min_cycles_between_rounds = 1
    error_memory = ErrorMemory(config.error_memory_path)
    for _ in range(3):
        error_memory.record("algebra_error", ["energy"], "bad algebra", "rederive_math", resolved=True)

    scheduler = Scheduler(config, dry_run=True)
    decisions = scheduler.run_cycle()

    assert not any(d.action == "solve" for d in decisions)  # queue was empty
    assert any(d.action == "practice" for d in decisions)


def test_run_loop_runs_bounded_number_of_cycles(config):
    calls = []

    def fake_sleep(seconds):
        calls.append(seconds)

    scheduler = Scheduler(config, dry_run=True)
    scheduler.run_loop(interval_seconds=0.01, max_cycles=3, sleep_fn=fake_sleep)

    assert scheduler.state.total_cycles == 3
    assert len(calls) == 2  # slept between cycles, not after the last one


def _seed_repeated_expression(config, n=5, expression="Eq(m*g*h, 0.5*m*v**2)", domain_tags=None):
    memory = EpisodicMemory(config.episodic_memory_path)
    output = json.dumps(
        {
            "expression": expression,
            "solve_for": "v",
            "substitutions": {},
            "solutions": ["9.9"],
            "solutions_numeric": [9.9],
        }
    )
    for _ in range(n):
        t = Trace.new("x")
        t.domain_tags = domain_tags or ["dynamics"]
        t.resolution_status = "passed_initial"
        t.tool_calls = [ToolCall(tool="symbolic_math", input="{}", output=output, latency_ms=5.0)]
        memory.write(t)


def test_growth_does_not_trigger_before_cooldown(config):
    config.scheduler_growth_min_cycles_between_rounds = 3
    _seed_repeated_expression(config)

    scheduler = Scheduler(config, dry_run=True)
    decisions = scheduler.run_cycle()  # cycles_since_last_growth becomes 1, below 3
    assert not any(d.action == "grow" for d in decisions)


def test_growth_triggers_and_adds_semantic_fact(config):
    config.scheduler_growth_min_cycles_between_rounds = 1
    _seed_repeated_expression(config, n=5)

    scheduler = Scheduler(config, dry_run=True)
    decisions = scheduler.run_cycle()

    grow_decisions = [d for d in decisions if d.action == "grow"]
    assert len(grow_decisions) == 1
    assert grow_decisions[0].details["n_facts_added"] == 1
    assert scheduler.state.total_growth_rounds == 1
    assert scheduler.state.cycles_since_last_growth == 0

    from physics_agent.retrieval import SemanticStore

    semantic = SemanticStore(config.semantic_store_path)
    assert len(semantic.entries) == 1
    assert semantic.entries[0]["confidence"] == 0.3
    assert "self_derived" in semantic.entries[0]["provenance"]


def test_growth_does_not_trigger_with_no_candidates(config):
    config.scheduler_growth_min_cycles_between_rounds = 1
    # No episodic history at all.
    scheduler = Scheduler(config, dry_run=True)
    decisions = scheduler.run_cycle()
    assert not any(d.action == "grow" for d in decisions)


def test_growth_does_not_redecide_an_already_proposed_candidate(config):
    config.scheduler_growth_min_cycles_between_rounds = 1
    _seed_repeated_expression(config, n=5)

    scheduler = Scheduler(config, dry_run=True)
    decisions1 = scheduler.run_cycle()
    assert any(d.action == "grow" for d in decisions1)
    assert scheduler.state.total_growth_rounds == 1

    # Cooldown was reset to 0 by the first round; force it past the
    # threshold again to simulate cycles passing, without adding any NEW
    # episodic data -- the same candidate is still the only one there.
    scheduler.state.cycles_since_last_growth = 5
    decisions2 = scheduler.run_cycle()

    assert not any(d.action == "grow" for d in decisions2)
    assert scheduler.state.total_growth_rounds == 1  # unchanged -- nothing new added

    from physics_agent.retrieval import SemanticStore

    semantic = SemanticStore(config.semantic_store_path)
    assert len(semantic.entries) == 1  # still just the one fact, not duplicated


def test_growth_can_fire_on_an_otherwise_idle_cycle(config):
    config.scheduler_growth_min_cycles_between_rounds = 1
    _seed_repeated_expression(config, n=5)

    scheduler = Scheduler(config, dry_run=True)
    decisions = scheduler.run_cycle()

    assert not any(d.action == "solve" for d in decisions)  # queue was empty
    assert any(d.action == "grow" for d in decisions)
