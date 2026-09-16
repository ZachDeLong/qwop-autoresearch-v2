import json

import pytest

from qwop_lab import artifacts


@pytest.mark.skipif(artifacts.os.name != "nt", reason="Windows reader file lock")
def test_atomic_writer_succeeds_after_real_reader_releases_lock(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    path.write_text('{"steps": 1}')
    observed = []
    with path.open(encoding="utf-8") as reader:

        def release_reader(seconds):
            observed.append(json.load(reader))
            reader.close()

        monkeypatch.setattr(artifacts.time, "sleep", release_reader)
        artifacts.write_json(path, {"steps": 2})
    assert observed == [{"steps": 1}]
    assert artifacts.read_json(path) == {"steps": 2}
    assert len(list(tmp_path.iterdir())) == 1


def test_atomic_writer_retries_transient_permission_errors(tmp_path, monkeypatch):
    replace = artifacts.os.replace
    calls = []

    def locked_twice(source, destination):
        calls.append(source)
        if len(calls) < 3:
            raise PermissionError("Windows reader holds destination")
        return replace(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", locked_twice)
    monkeypatch.setattr(artifacts.time, "sleep", lambda seconds: None)
    artifacts.write_json(tmp_path / "progress.json", {"steps": 8192})
    assert len(calls) == 3
    assert artifacts.read_json(tmp_path / "progress.json") == {"steps": 8192}
    assert len(list(tmp_path.iterdir())) == 1


def test_persistent_lock_keeps_previous_artifact_and_surfaces_error(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    path.write_text('{"steps": 8192}')
    calls = []

    def locked(source, destination):
        calls.append(source)
        raise PermissionError("persistent lock")

    monkeypatch.setattr(artifacts.os, "replace", locked)
    monkeypatch.setattr(artifacts.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        artifacts.write_json(path, {"steps": 16384})
    assert len(calls) == 10
    assert artifacts.read_json(path) == {"steps": 8192}
    assert len(list(tmp_path.iterdir())) == 1
