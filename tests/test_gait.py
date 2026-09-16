import math
from types import SimpleNamespace

import numpy as np
import pytest

from qwop_lab.bootstrap import match_locked_newlines
from qwop_lab.artifacts import sha256
from qwop_lab.gait import (
    GROUND_Y,
    HIP_LEFT,
    HIP_RIGHT,
    KNEE_LEFT,
    KNEE_RIGHT,
    NECK,
    GaitConfig,
    GaitWrapper,
    measure,
    reward_adjustment,
)
from qwop_lab.replay import verify_sample
from test_integrity import ToyEnv


def upright_pose():
    pose = np.zeros((12, 3))
    hip = (np.array(HIP_LEFT) + HIP_RIGHT) / 2
    axis = np.array(NECK) - hip
    angle = -math.pi / 2 - math.atan2(axis[1], axis[0])
    pose[0, 2] = angle
    pose[0, 1] = GROUND_Y - 8 - (math.sin(angle) * hip[0] + math.cos(angle) * hip[1])
    pose[3, 1] = GROUND_Y - 3 - KNEE_LEFT[1]
    pose[8, 1] = GROUND_Y - 3 - KNEE_RIGHT[1]
    return pose


def test_geometry_uses_joints_y_down_and_periodic_angles():
    pose = upright_pose()
    metrics = measure(pose, GaitConfig())
    assert metrics["pelvis_height_m"] == pytest.approx(0.8)
    assert metrics["torso_tilt_degrees"] == pytest.approx(0, abs=1e-10)
    assert metrics["left_knee_clearance_m"] == pytest.approx(0.3)
    assert metrics["upright"]
    pose[:, 0] += 1000
    pose[:, 2] += 4 * math.pi
    periodic = measure(pose, GaitConfig())
    assert periodic["posture_quality"] == pytest.approx(metrics["posture_quality"])
    assert periodic["upright"]
    pose[3, 1] += 2.9
    assert measure(pose, GaitConfig())["knee_near_ground"]
    assert not measure(pose, GaitConfig())["upright"]


def test_reward_cannot_pay_for_standing_or_backwards_motion():
    for quality in (0, 0.5, 1):
        assert reward_adjustment(0, quality, 1) == 0
        assert reward_adjustment(-1, quality, 1) == 0
        assert reward_adjustment(1, quality, 0) == 0
    assert 1 + reward_adjustment(1, 0, 1) == pytest.approx(0.1)
    assert 1 + reward_adjustment(1, 1, 1) == pytest.approx(2)


class RawToyEnv(ToyEnv):
    def reset(self, **kwargs):
        self.raw_observation = np.zeros((12, 5), np.float32)
        self.raw_observation[:, :3] = upright_pose()
        self.last_reaction = SimpleNamespace(time=0, distance=0)
        return super().reset(**kwargs)

    def step(self, action):
        transition = super().step(action)
        self.last_reaction = SimpleNamespace(**{k: transition[4][k] for k in ("time", "distance")})
        return transition


def test_control_preserves_observations_rewards_flags_and_resets_statistics():
    env = GaitWrapper(RawToyEnv(), GaitConfig())
    reference = RawToyEnv()
    for _ in range(2):
        np.testing.assert_array_equal(env.reset()[0], reference.reset()[0])
        for _ in range(3):
            actual, expected = env.step(0), reference.step(0)
            np.testing.assert_array_equal(actual[0], expected[0])
            assert actual[1:4] == expected[1:4]
            for key, value in expected[4].items():
                assert actual[4][key] == value
        summary = actual[4]["gait_episode"]
        assert summary["steps"] == 3
        assert summary["gait_adjustment_return"] == 0
        assert summary["upright_forward_fraction"] == 1


def test_replay_checks_raw_geometry_even_when_clipped_observation_matches():
    state = {
        "obs_sha256": "same",
        "distance": 0,
        "terminated": False,
        "truncated": False,
        "is_success": False,
        "raw_time": 0,
        "gait_pose": [[1, 2, 3]],
        "gait": {"upright": True},
    }
    with pytest.raises(ValueError, match="gait_pose"):
        verify_sample({**state, "gait_pose": [[1, 2, 30]]}, state, 0)


def test_newline_portability_preserves_hash_guard(tmp_path):
    source, locked = tmp_path / "source.js", tmp_path / "locked.js"
    source.write_bytes(b"hello\nworld\n")
    locked.write_bytes(b"hello\r\nworld\r\n")
    match_locked_newlines(source, sha256(locked))
    assert source.read_bytes() == locked.read_bytes()
    source.write_bytes(b"changed\n")
    with pytest.raises(RuntimeError, match="differs"):
        match_locked_newlines(source, sha256(locked))


def test_long_episode_replay_allows_only_float32_rounding_not_clock_drift():
    initial_a, initial_b = np.float32(0.003), np.float32(0.003219)
    offset = float(initial_b) - float(initial_a)
    for elapsed in (0.1, 20.0, 63.99, 64.0, 66.66):
        expected = {
            "obs_sha256": "same",
            "distance": 1,
            "terminated": False,
            "truncated": False,
            "is_success": False,
            "raw_time": float(np.float32(float(initial_a) + elapsed)),
        }
        actual = {**expected, "raw_time": float(np.float32(float(initial_b) + elapsed))}
        verify_sample(actual, expected, 1, offset)
        with pytest.raises(ValueError, match="raw_time"):
            verify_sample({**actual, "raw_time": actual["raw_time"] + 0.0001}, expected, 1, offset)
