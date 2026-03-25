"""
baseline/agent.py — Eval harness adapter for the v1 baseline model.

Loads the vanilla CleanRL PPO model from v1 and exposes get_action(obs).
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

# The v1 network architecture (must match exactly)
OBS_DIM = 60
ACT_DIM = 16


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(OBS_DIM, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(OBS_DIM, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, ACT_DIM),
        )
        for layer in [*self.critic, *self.actor]:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


# Load model weights
_device = torch.device("cpu")
_agent = Agent().to(_device)

# Try to find the v1 model weights
_v1_model_path = os.path.join(os.path.dirname(__file__), "..", "..", "qwop-autoresearch", "model.pt")
_v1_baseline_model_path = os.path.join(os.path.dirname(__file__), "..", "..", "qwop-autoresearch", "baseline", "results", "model.pt")

_model_path = None
if os.path.exists(_v1_baseline_model_path):
    _model_path = _v1_baseline_model_path
elif os.path.exists(_v1_model_path):
    _model_path = _v1_model_path

if _model_path:
    _agent.load_state_dict(torch.load(_model_path, map_location=_device, weights_only=True))
    _agent.eval()
    print(f"Loaded baseline model from: {_model_path}")
else:
    print(f"WARNING: No model weights found. Looked in:")
    print(f"  {_v1_baseline_model_path}")
    print(f"  {_v1_model_path}")
    print(f"Agent will use random (untrained) weights.")


def get_action(obs: np.ndarray) -> int:
    """Given a 60-float observation, return an action in [0, 15]."""
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(_device)
        action, _, _, _ = _agent.get_action_and_value(obs_tensor)
        return int(action.item())
