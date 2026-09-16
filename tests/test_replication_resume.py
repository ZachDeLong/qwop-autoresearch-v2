from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from qwop_lab import replication_resume as recovery
from qwop_lab.artifacts import read_json, sha256, write_json
from qwop_lab.budget import Ledger
from qwop_lab.budget import StepMeter
from qwop_lab.training import FRESH_PPO_CONFIG
from test_integrity import ToyEnv


def test_resume_retains_spent_and_discarded_rollout_steps(tmp_path, monkeypatch):
    protocol = {
        "seeds": [17, 29, 61],
        "designs": {"baseline": {}, "kinematic": {}},
        "protected_sources": {},
        "training_steps_per_run": 128,
        "ppo_config": {"n_steps": 32},
    }
    write_json(tmp_path / "protocol.json", protocol)
    (tmp_path / "protocol.sha256").write_text(sha256(tmp_path / "protocol.json"))
    for seed in protocol["seeds"]:
        for design in protocol["designs"]:
            write_json(tmp_path / design / f"seed-{seed}/result.json", {"status": "complete"})
    path = tmp_path / "baseline/seed-29"
    write_json(
        path / "result.json", {"status": "failed", "error": "PermissionError: progress.json"}
    )
    failed_bytes = (path / "result.json").read_bytes()
    write_json(path / "training/manifest.json", {"status": "failed", "actual_steps": 64})
    (path / "training/interrupted.zip").write_bytes(b"saved weights")
    ledger = Ledger(path / "training.sqlite", 128)
    lease = ledger.lease("original")
    for _ in range(64):
        lease.consume()
    lease.close()
    monkeypatch.setattr(
        recovery.PPO,
        "load",
        Mock(
            return_value=SimpleNamespace(
                num_timesteps=64,
                _n_updates=2,
                n_epochs=2,
                n_steps=32,
            )
        ),
    )
    amendment = recovery.amend(tmp_path)
    assert amendment["spent_training_steps"] == 64
    assert amendment["remaining_training_steps"] == 64
    assert amendment["discarded_rollout_interactions"] == 32
    assert ledger.status()["cap"] == 128
    assert (path / "result.json").read_bytes() == failed_bytes
    assert read_json(tmp_path / "resume-amendment.json") == amendment


def test_progress_retry_does_not_lose_artifact_or_retry_forever(tmp_path, monkeypatch):
    original = recovery.write_json_once
    attempts = []

    def transient(path, value):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("reader holds file")
        original(path, value)

    monkeypatch.setattr(recovery, "write_json_once", transient)
    monkeypatch.setattr(recovery.time, "sleep", lambda seconds: None)
    recovery.write_json(tmp_path / "progress.json", {"steps": 64})
    assert len(attempts) == 3
    assert read_json(tmp_path / "progress.json") == {"steps": 64}
    blocked = Mock(side_effect=PermissionError("persistent lock"))
    monkeypatch.setattr(recovery, "write_json_once", blocked)
    with pytest.raises(PermissionError):
        recovery.write_json(tmp_path / "progress.json", {"steps": 128})
    assert blocked.call_count == 10
    assert read_json(tmp_path / "progress.json") == {"steps": 64}


def test_continuation_reaches_total_budget_without_retraining_spent_steps(tmp_path, monkeypatch):
    recovery.configure_torch(29)
    path = tmp_path / "baseline/seed-29"
    (path / "training").mkdir(parents=True)
    config = {**FRESH_PPO_CONFIG, "n_steps": 32, "batch_size": 16, "n_epochs": 2}
    ledger = Ledger(path / "training.sqlite", 128)
    lease = ledger.lease("interrupted")
    env = StepMeter(ToyEnv(), lease)
    model = PPO(
        "MlpPolicy",
        env,
        seed=29,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        **config,
    )

    class StopBeforeUpdate(BaseCallback):
        def _on_step(self):
            return self.num_timesteps < 64

    model.learn(128, callback=StopBeforeUpdate())
    model.save(path / "training/interrupted.zip")
    env.close()
    assert model.num_timesteps == ledger.status()["charged_steps"] == 64
    assert model._n_updates == 2
    write_json(path / "training/manifest.json", {"wall_seconds": 0})
    protocol = {
        "designs": {"baseline": {"layers": [128, 128]}},
        "max_workers": 3,
        "ppo_config": config,
        "contract": {"id": "test"},
        "checkpoint_interval": 128,
        "training_wall_seconds_per_run": 120,
        "training_steps_per_run": 128,
    }
    amendment = {
        "design": "baseline",
        "seed": 29,
        "remaining_training_steps": 64,
        "spent_training_steps": 64,
        "interrupted_checkpoint_sha256": sha256(path / "training/interrupted.zip"),
    }
    write_json(tmp_path / "resume-amendment.json", amendment)
    monkeypatch.setattr(recovery, "verify", lambda folder: (protocol, amendment))
    monkeypatch.setattr(recovery, "make_env", lambda lease: StepMeter(ToyEnv(), lease))
    monkeypatch.setattr(recovery, "evaluate_recovered", lambda folder: {"ready": True})
    monkeypatch.setattr(recovery.training, "write_json", recovery.training.write_json)
    assert recovery.resume(tmp_path) == {"ready": True}
    saved = PPO.load(path / "recovery-training/final.zip")
    assert ledger.status()["charged_steps"] == saved.num_timesteps == 128
    assert saved._n_updates == 6  # One interrupted rollout is charged but never updated.
    assert read_json(path / "recovery-training/manifest.json")["actual_steps"] == 64
