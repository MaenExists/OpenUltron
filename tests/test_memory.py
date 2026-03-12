from openultron import memory


def test_append_experience(tmp_path, monkeypatch):
    experiences_dir = tmp_path / "experiences"
    monkeypatch.setattr(memory, "EXPERIENCES_DIR", experiences_dir)

    path = memory.append_experience("Test entry")
    assert path.exists()

    entries = memory.latest_experience_entries(count=1)
    assert entries
    assert "Test entry" in entries[-1]["body"]
