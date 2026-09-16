import pytest

from qwop_lab import campaign_recovery as recovery
from qwop_lab.artifacts import read_json, write_json


def test_recovery_preserves_failed_result_and_does_not_expand_training_allowance(
    tmp_path, monkeypatch
):
    protocol = {
        "arms": {"research": {"proposal-1": 524288}},
        "ppo_config": {"n_steps": 2048},
    }
    monkeypatch.setattr(recovery, "verify_protocol", lambda folder: protocol)
    write_json(tmp_path / "protocol.json", protocol)
    failed = tmp_path / "research/proposal-1/result.json"
    write_json(failed, {"status": "failed", "error": "RuntimeError: Ambiguous game response"})
    original_bytes = failed.read_bytes()
    write_json(tmp_path / "research/proposal-1/training/manifest.json", {"actual_steps": 46253})
    write_json(
        tmp_path / "research/proposal-1/proposal.json", {"architecture": {"layers": [64, 64]}}
    )
    (tmp_path / "source/qwop_lab").mkdir(parents=True)
    amendment = recovery.amend(tmp_path)
    assert amendment["retry_steps"] == 477184
    assert (
        amendment["failed_attempt_charged_steps"]
        + amendment["retry_steps"]
        + amendment["unused_rounding_steps"]
        == 524288
    )
    assert failed.read_bytes() == original_bytes
    recovery.verified_amendment(tmp_path)
    with pytest.raises(FileExistsError):
        recovery.amend(tmp_path)
    data = read_json(tmp_path / "amendment-001.json")
    data["retry_steps"] += 2048
    write_json(tmp_path / "amendment-001.json", data)
    with pytest.raises(ValueError, match="changed"):
        recovery.verified_amendment(tmp_path)
