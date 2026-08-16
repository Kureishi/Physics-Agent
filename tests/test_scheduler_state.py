from physics_agent.scheduler.state import SchedulerState, load_state, save_state


def test_load_state_defaults_when_missing(tmp_path):
    state = load_state(tmp_path / "state.json")
    assert state == SchedulerState()
    assert state.total_cycles == 0


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    state = SchedulerState(
        total_cycles=5,
        total_solves=3,
        total_reviews=1,
        total_curriculum_rounds=1,
        solves_since_last_review=2,
        cycles_since_last_curriculum=4,
    )
    save_state(path, state)

    reloaded = load_state(path)
    assert reloaded == state


def test_load_state_empty_file_returns_defaults(tmp_path):
    path = tmp_path / "state.json"
    path.touch()
    state = load_state(path)
    assert state == SchedulerState()
