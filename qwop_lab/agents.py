"""Explicit model loading; legacy models never fall back to random weights."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .artifacts import sha256


def configure_torch(seed=1):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


class LegacyPolicy:
    def __init__(self, checkpoint):
        self.path = Path(checkpoint).resolve()
        weights = torch.load(self.path, map_location="cpu", weights_only=True)
        width = weights["actor.0.weight"].shape[0]
        self.actor = nn.Sequential(
            nn.Linear(60, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 16),
        )
        self.actor.load_state_dict(
            {k.removeprefix("actor."): v for k, v in weights.items() if k.startswith("actor.")},
            strict=True,
        )
        self.actor.eval()

    def __call__(self, obs):
        with torch.inference_mode():
            return int(self.actor(torch.as_tensor(obs)).argmax().item())


class SB3Policy:
    def __init__(self, checkpoint):
        from stable_baselines3 import PPO

        self.path = Path(checkpoint).resolve()
        self.model = PPO.load(self.path, device="cpu")

    def __call__(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def load_policy(checkpoint):
    configure_torch()
    path = Path(checkpoint).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint missing: {path}")
    policy = SB3Policy(path) if path.suffix == ".zip" else LegacyPolicy(path)
    identity = {
        "path": str(path),
        "sha256": sha256(path),
        "format": "sb3-ppo" if path.suffix == ".zip" else "legacy-tanh-ppo",
        "action_mode": "argmax",
    }
    return policy, identity


def import_legacy_weights(model, checkpoint):
    """Map the old independent actor/critic MLPs to SB3 without changing logits."""
    old = torch.load(checkpoint, map_location="cpu", weights_only=True)
    new = model.policy.state_dict()
    for source, target in (("actor", "policy"), ("critic", "value")):
        for index in (0, 2):
            for param in ("weight", "bias"):
                new[f"mlp_extractor.{target}_net.{index}.{param}"] = old[
                    f"{source}.{index}.{param}"
                ]
        head = "action_net" if source == "actor" else "value_net"
        for param in ("weight", "bias"):
            new[f"{head}.{param}"] = old[f"{source}.4.{param}"]
    model.policy.load_state_dict(new, strict=True)
