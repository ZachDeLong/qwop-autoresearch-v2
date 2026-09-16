import copy

import pytest

from qwop_lab import gait_study
from qwop_lab.artifacts import digest, read_json, write_json
from qwop_lab.budget import Ledger
from qwop_lab.environment import Case
from qwop_lab.evaluation import rollout
from qwop_lab.gait import GaitConfig, GaitWrapper
from test_gait import RawToyEnv


def test_audit_recomputes_geometry_and_summary_even_after_rehashing():
    episode = rollout(GaitWrapper(RawToyEnv(), GaitConfig()), lambda obs: 0, Case(101))
    gait_study.audit_episode(episode, GaitConfig())
    tampered = copy.deepcopy(episode)
    tampered["trace"][0]["gait_pose"][0][1] += 5
    tampered["trace_sha256"] = digest(tampered["trace"])
    with pytest.raises(ValueError, match="measurements"):
        gait_study.audit_episode(tampered, GaitConfig())
    episode["gait_summary"]["mean_upright"] = 0
    with pytest.raises(ValueError, match="summary"):
        gait_study.audit_episode(episode, GaitConfig())


def init_test(tmp_path, monkeypatch):
    monkeypatch.setattr(gait_study, "runtime", lambda: {"runtime_id": "test"})
    monkeypatch.setattr(gait_study.subprocess, "check_output", lambda *args, **kwargs: "test\n")
    folder = tmp_path / "study"
    protocol = gait_study.initialize(folder, steps=4096, interval=2048)
    return folder, protocol


def test_frozen_protocol_and_worst_case_budget(tmp_path, monkeypatch):
    folder, protocol = init_test(tmp_path, monkeypatch)
    assert protocol["evaluation_cap_per_arm"] == 25000  # Four cases plus final replay.
    gait_study.verify_protocol(folder)
    protocol["seed"] += 1
    write_json(folder / "protocol.json", protocol)
    with pytest.raises(ValueError, match="protocol changed"):
        gait_study.verify_protocol(folder)


def test_failed_training_is_preserved_and_cannot_acquire_a_retry(tmp_path, monkeypatch):
    folder, _ = init_test(tmp_path, monkeypatch)

    def fail(checkpoint, out_dir, ledger, **kwargs):
        lease = ledger.lease("failed training")
        lease.consume()
        lease.consume()
        lease.close()
        raise RuntimeError("transport failed")

    monkeypatch.setattr(gait_study, "train", fail)
    with pytest.raises(RuntimeError, match="transport failed"):
        gait_study.run_trial(folder, "gait")
    result = read_json(folder / "gait/result.json")
    assert result["status"] == "failed"
    assert result["training_budget"]["confirmed_steps"] == 2
    with pytest.raises(FileExistsError):
        gait_study.run_trial(folder, "gait")
    assert Ledger(folder / "gait/training.sqlite").status()["confirmed_steps"] == 2
