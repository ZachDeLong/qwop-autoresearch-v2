import json

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from qwop_lab.agents import load_policy
from qwop_lab.architectures import Architecture
from qwop_lab.artifacts import read_json, sha256, write_json
from qwop_lab.budget import Ledger
from qwop_lab.campaign import CONVENTIONAL_LAYERS, selection_key, validate_proposal
from qwop_lab.candidates.body_grouped_v1 import BodyGroupedExtractor
from qwop_lab.candidates.kinematic_v2 import KinematicExtractor
from qwop_lab.training import FRESH_PPO_CONFIG, train
from test_integrity import ToyEnv


def test_leg_encoder_uses_correct_parts_and_local_coordinate_scale():
    extractor = BodyGroupedExtractor(ToyEnv.observation_space)
    observations = torch.zeros(1, 12, 5)
    observations[0, 0] = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6])
    for index in (6, 3, 4, 11, 8, 9):
        observations[0, index, 0] = 0.2 + index / 530
        observations[0, index, 1] = 0.3 + index / 10
    legs = extractor.leg_inputs(observations.flatten(start_dim=1))
    torch.testing.assert_close(legs[0, 0, [0, 5, 10]], torch.tensor([0.6, 0.3, 0.4]))
    torch.testing.assert_close(legs[0, 1, [0, 5, 10]], torch.tensor([1.1, 0.8, 0.9]))
    torch.testing.assert_close(legs[0, 0, -5:], observations[0, 0])
    assert extractor(observations.flatten(start_dim=1)).shape == (1, 128)


@pytest.mark.parametrize(
    "extractor",
    [
        "qwop_lab.candidates.body_grouped_v1:BodyGroupedExtractor",
        "qwop_lab.candidates.kinematic_v2:KinematicExtractor",
    ],
)
def test_custom_architecture_trains_and_reloads_exactly_with_source_identity(
    tmp_path, monkeypatch, extractor
):
    import qwop_lab.training as training

    monkeypatch.setattr(training, "runtime", lambda: {"runtime_id": "test"})
    monkeypatch.setattr(
        training,
        "make_env",
        lambda lease, **kwargs: __import__("qwop_lab.budget", fromlist=["StepMeter"]).StepMeter(
            ToyEnv(), lease
        ),
    )
    design = Architecture((64, 64), extractor=extractor)
    ledger = Ledger(tmp_path / "budget.sqlite", 64)
    checkpoint = train(
        None,
        tmp_path / "train",
        ledger,
        steps=64,
        architecture=design,
        config={**FRESH_PPO_CONFIG, "n_steps": 32, "batch_size": 16, "n_epochs": 2},
        checkpoint_interval=32,
    )
    manifest = read_json(tmp_path / "train" / "manifest.json")
    assert manifest["actual_steps"] == 64
    assert 0 < manifest["trainable_parameters"] <= 250000
    assert ledger.status()["confirmed_steps"] == 64
    middle = tmp_path / "train" / "checkpoints" / "step-000000032.zip"
    assert PPO.load(middle).num_timesteps == 32
    final = PPO.load(checkpoint)
    final_copy = PPO.load(tmp_path / "train" / "checkpoints" / "step-000000064.zip")
    for key, value in final.policy.state_dict().items():
        torch.testing.assert_close(value, final_copy.policy.state_dict()[key], rtol=0, atol=0)
    policy, identity = load_policy(checkpoint)
    obs = np.zeros(60, np.float32)
    assert policy(obs) == int(final.predict(obs, deterministic=True)[0])
    assert identity["sha256"] == sha256(checkpoint)
    sidecar = checkpoint.with_suffix(".json")
    data = read_json(sidecar)
    data["architecture"]["source_sha256"] = "tampered"
    write_json(sidecar, data)
    with pytest.raises(ValueError, match="source changed"):
        load_policy(checkpoint)


def test_proposal_respects_conventional_design_and_requires_actual_prior_evidence(tmp_path):
    protocol = {
        "arms": {"search": {"small": 10}, "research": {"proposal-2": 10}},
        "conventional_layers": CONVENTIONAL_LAYERS,
    }
    proposal = {
        "arm": "search",
        "slot": "small",
        "architecture": {"layers": [64, 64]},
        "hypothesis": "Compare width",
        "expected_signal": "Faster finish",
        "evidence": [],
    }
    assert validate_proposal(tmp_path, proposal, protocol).layers == (64, 64)
    with pytest.raises(ValueError, match="fixed"):
        validate_proposal(tmp_path, {**proposal, "architecture": {"layers": [128, 128]}}, protocol)
    prior = tmp_path / "research" / "proposal-1" / "result.json"
    write_json(prior, {"status": "complete"})
    proposal.update(arm="research", slot="proposal-2")
    with pytest.raises(ValueError, match="cite"):
        validate_proposal(tmp_path, proposal, protocol)
    proposal["evidence"] = [{"path": "research/proposal-1/result.json", "sha256": sha256(prior)}]
    validate_proposal(tmp_path, proposal, protocol)
    prior.write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="changed"):
        validate_proposal(tmp_path, proposal, protocol)


def test_selection_never_treats_failure_as_fast_finish():
    slow = {"best_100m_game_seconds": 20, "valid_100m_finish_rate": 1, "mean_distance": 100}
    fast = {"best_100m_game_seconds": 15, "valid_100m_finish_rate": 0.5, "mean_distance": 60}
    failure = {"best_100m_game_seconds": None, "valid_100m_finish_rate": 0, "mean_distance": 99}
    assert selection_key(fast) < selection_key(slow) < selection_key(failure)


def test_kinematic_features_preserve_raw_state_and_encode_relative_motion():
    encoder = KinematicExtractor(ToyEnv.observation_space)
    assert sum(p.numel() for p in encoder.parameters()) == 0
    obs = torch.zeros(1, 12, 5)
    obs[0, 3] = torch.tensor([0.01, 0.2, torch.pi / 12, 0.3, 0.4])
    out = encoder(obs.flatten(start_dim=1))
    torch.testing.assert_close(out[:, :60], obs.flatten(start_dim=1), rtol=0, atol=0)
    features = out[:, 60:].reshape(1, 12, 6)
    torch.testing.assert_close(features[0, 0], torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(
        features[0, 3], torch.tensor([0.53, 0.2, 1.0, 0.0, 1.2, 1.7]), atol=1e-6, rtol=1e-6
    )
