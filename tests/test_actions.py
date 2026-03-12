from openultron import actions


def test_queue_roundtrip(tmp_path, monkeypatch):
    queue_file = tmp_path / "actions_queue.md"
    log_dir = tmp_path / "actions"
    monkeypatch.setattr(actions, "QUEUE_FILE", queue_file)
    monkeypatch.setattr(actions, "ACTION_LOG_DIR", log_dir)

    action = actions.Action(
        id="abc123",
        created_at="2026-03-12 00:00:00 UTC",
        status="proposed",
        type="shell",
        title="List files",
        payload={"cmd": "ls"},
    )
    actions.save_queue([action])
    loaded = actions.load_queue()

    assert len(loaded) == 1
    assert loaded[0].id == "abc123"
    assert loaded[0].type == "shell"


def test_update_action(tmp_path, monkeypatch):
    queue_file = tmp_path / "actions_queue.md"
    monkeypatch.setattr(actions, "QUEUE_FILE", queue_file)

    action = actions.Action(
        id="xyz789",
        created_at="2026-03-12 00:00:00 UTC",
        status="proposed",
        type="write_memory",
        title="Write memo",
        payload={"path": "knowledge/test.md", "content": "Hello"},
    )
    actions.save_queue([action])
    updated = actions.update_action("xyz789", status="approved")

    assert updated is not None
    assert updated.status == "approved"
