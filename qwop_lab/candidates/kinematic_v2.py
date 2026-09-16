"""Adaptive second proposal: explicit kinematics without a learned shared encoder.

Preserves every raw observation and adds torso-relative positions/velocities plus
sine/cosine of relative angles. These are standard engineered features, not a
claim of a novel learning algorithm. Constants invert the locked runtime's scales.
"""

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class KinematicExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        if observation_space.shape != (60,):
            raise ValueError("Expected the locked QWOP observation")
        super().__init__(observation_space, features_dim=132)

    def forward(self, observations):
        bodies = observations.reshape(-1, 12, 5)
        relative = bodies - bodies[:, :1]
        angles = relative[..., 2] * 6
        kinematics = torch.stack(
            (
                relative[..., 0] * 53,  # position x: 530 world units -> 10
                relative[..., 1],  # position y: already scaled by 10
                torch.sin(angles),
                torch.cos(angles),
                relative[..., 3] * 4,  # velocity x: 40 -> 10
                relative[..., 4] * 4.25,  # velocity y: 42.5 -> 10
            ),
            dim=-1,
        )
        return torch.cat((observations, kinematics.flatten(start_dim=1)), dim=1)
