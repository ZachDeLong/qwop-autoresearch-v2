import pytest

from qwop_lab import replication
from qwop_lab.artifacts import read_json, sha256, write_json


def result(finishes, mean_time, distance):
    return {
        "final_summary": {
            "valid_100m_finishes": finishes,
            "episodes": 2,
            "best_100m_game_seconds": mean_time,
            "mean_100m_game_seconds": mean_time,
            "mean_distance": distance,
        }
    }


def test_replication_aggregates_training_seeds_without_hiding_failures():
    rows = {
        ("baseline", 17): result(2, 10, 101),
        ("kinematic", 17): result(2, 9, 101),
        ("baseline", 29): result(2, 12, 101),
        ("kinematic", 29): result(1, 4, 52),
        ("baseline", 61): result(0, None, 1),
        ("kinematic", 61): result(2, 11, 101),
    }
    summary = replication.aggregate(rows, [17, 29, 61])
    assert summary["comparable_speed_pairs"] == 1
    assert summary["mean_paired_time_change_percent"] == pytest.approx(-10)
    assert summary["designs"]["kinematic"]["phase_finishes"] == 5
    assert summary["designs"]["kinematic"]["runs_finishing_both_phases"] == 2
    assert summary["designs"]["kinematic"]["mean_time_among_both_phase_finishing_runs"] == 10
    assert summary["pairs"][1]["kinematic"]["conditional_mean_100m_game_seconds"] == 4
    assert summary["pairs"][1]["kinematic_time_change_percent"] is None


def test_replication_refuses_changed_protocol_and_sources(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    source = root / "source.py"
    source.write_text("original")
    monkeypatch.setattr(replication, "ROOT", root)
    monkeypatch.setattr(replication, "runtime", lambda: {"runtime_id": "test"})
    protocol = {
        "protected_sources": {"source.py": sha256(source)},
        "contract": replication.contract({"runtime_id": "test"}),
    }
    write_json(tmp_path / "protocol.json", protocol)
    (tmp_path / "protocol.sha256").write_text(sha256(tmp_path / "protocol.json"))
    replication.verify_protocol(tmp_path)
    source.write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        replication.verify_protocol(tmp_path)
    write_json(tmp_path / "protocol.json", {})
    with pytest.raises(ValueError, match="protocol changed"):
        replication.verify_protocol(tmp_path)


def test_failed_training_cannot_silently_restart_or_expand_budget(tmp_path, monkeypatch):
    protocol = {
        "designs": replication.DESIGNS,
        "seeds": [17],
        "training_steps_per_run": 1048576,
        "checkpoint_interval": 262144,
    }
    monkeypatch.setattr(replication, "verify_protocol", lambda folder: protocol)
    write_json(tmp_path / "protocol.json", protocol)
    path = tmp_path / "baseline/seed-17"
    ledger = replication.Ledger(path / "training.sqlite", 1048576)
    lease = ledger.lease("failed attempt")
    for _ in range(7):
        lease.consume()
    lease.close()
    replication.Ledger(path / "evaluation.sqlite", 50000)
    write_json(path / "training/manifest.json", {"status": "failed", "actual_steps": 7})
    monkeypatch.setattr(replication, "train", lambda *a, **kw: pytest.fail("Must not restart"))
    with pytest.raises(ValueError, match="restart is forbidden"):
        replication.run_trial(tmp_path, "baseline", 17)
    assert ledger.status()["charged_steps"] == 7
    assert ledger.status()["cap"] == 1048576
    assert read_json(path / "result.json")["status"] == "failed"
    assert not (path / "running.lock").exists()


def test_startup_replacement_only_accepts_zero_spent_steps(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "qwop_lab").mkdir(parents=True)
    (root / "qwop_lab/module.py").write_text("# frozen source")
    (root / "results/architecture-001").mkdir(parents=True)
    (root / "results/architecture-001/report.md").write_text("Prior evidence")
    monkeypatch.setattr(replication, "ROOT", root)
    monkeypatch.setattr(replication, "runtime", lambda: {"runtime_id": "test"})
    previous = tmp_path / "previous"
    replication.initialize(previous)
    for seed in replication.SEEDS:
        for design in replication.DESIGNS:
            write_json(previous / design / f"seed-{seed}/result.json", {"status": "failed"})
    recovered = replication.initialize(tmp_path / "replacement", previous)
    assert recovered["startup_recovery"]["previous_charged_steps"] == 0
    assert recovered["training_steps_per_run"] == 1048576
    lease = replication.Ledger(previous / "baseline/seed-17/training.sqlite").lease("one step")
    lease.consume()
    lease.close()
    with pytest.raises(ValueError, match="cannot discard any spent"):
        replication.initialize(tmp_path / "invalid-replacement", previous)


def test_batch_surfaces_failed_worker_after_preserving_outcomes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    protocol = {"seeds": [17], "designs": {"baseline": {}, "kinematic": {}}, "max_workers": 2}
    monkeypatch.setattr(replication, "verify_protocol", lambda folder: protocol)
    monkeypatch.setattr(
        replication.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1)
    )
    for design in protocol["designs"]:
        (tmp_path / design / "seed-17").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="failed workers"):
        replication.run_batch(tmp_path)
    outcomes = read_json(tmp_path / "batch.json")["outcomes"]
    assert len(outcomes) == 2 and all(row["exit_code"] == 1 for row in outcomes)
