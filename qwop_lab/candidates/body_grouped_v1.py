"""First hypothesis: share a leg encoder and expose body-relative limb positions.

This combines established architectural ideas; it is not a claim of scientific novelty.
Input ordering and normalization come from the locked qwop-gym 1.0.1 runtime.
The raw observation branch retains all information supplied to the conventional MLP.
"""

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class BodyGroupedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        if observation_space.shape != (60,):
            raise ValueError("Expected the locked 12-body, 5-feature observation")
        super().__init__(observation_space, features_dim=128)
        self.global_encoder = nn.Sequential(nn.Linear(60, 64), nn.Tanh())
        self.leg_encoder = nn.Sequential(nn.Linear(20, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        # thigh, calf, foot, as indexed by extensions.js OBS_PARTS.
        self.register_buffer("leg_indices", torch.tensor([[6, 3, 4], [11, 8, 9]]))

    def leg_inputs(self, observations):
        bodies = observations.reshape(-1, 12, 5)
        torso = bodies[:, 0]
        legs = bodies[:, self.leg_indices].clone()
        # x is normalized by 530 world units, y by 10. Use equal local scales.
        legs[..., 0] = (legs[..., 0] - torso[:, None, None, 0]) * 53
        legs[..., 1] = legs[..., 1] - torso[:, None, None, 1]
        return torch.cat((legs.flatten(start_dim=2), torso[:, None, :].expand(-1, 2, -1)), dim=2)

    def forward(self, observations):
        legs = self.leg_encoder(self.leg_inputs(observations)).flatten(start_dim=1)
        return torch.cat((self.global_encoder(observations), legs), dim=1)
