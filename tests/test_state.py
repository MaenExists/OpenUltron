from openultron import state


def test_state_roundtrip(tmp_path, monkeypatch):
    state_file = tmp_path / "state.md"
    monkeypatch.setattr(state, "STATE_FILE", state_file)

    initial = state.read_state()
    assert initial["status"] == "paused"

    updated = state.update_state(status="running", loop_count="3")
    assert updated["status"] == "running"
    assert updated["loop_count"] == "3"

    loaded = state.read_state()
    assert loaded["status"] == "running"
