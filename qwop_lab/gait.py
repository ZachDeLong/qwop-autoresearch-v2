"""Geometric gait diagnostics and a forward-motion-gated posture reward.

Landmarks are reconstructed from the locked game's body origins and local joint
anchors, not from sprite centres or clipped policy observations. Coordinates are
y-down; ten world units equal one displayed metre. Knee clearance is a geometric
proxy, not a contact sensor. These metrics do not establish alternating footfalls.
"""

from dataclasses import asdict, dataclass
import math

import gymnasium as gym
import numpy as np

# Track centre minus half its 64-pixel texture height / worldScale (20).
GROUND_Y = 10.74275 - 64 / 40
TORSO_INITIAL = (2.5111726226000157, -1.8709517533957938, -1.2514497119301329)
LEFT_CALF_INITIAL = (3.12585731974087, 5.525511655361298, -1.5903971528225265)
RIGHT_CALF_INITIAL = (-0.07253905736790486, 5.347881871063159, -0.7588859967104447)


def local_anchor(initial, point):
    x, y, angle = initial
    dx, dy = point[0] - x, point[1] - y
    c, s = math.cos(angle), math.sin(angle)
    return c * dx + s * dy, -s * dx + c * dy


HIP_LEFT = local_anchor(TORSO_INITIAL, (2.003367181376716, 0.23802590387419476))
HIP_RIGHT = local_anchor(TORSO_INITIAL, (1.2470052823973599, -0.011635347168778898))
NECK = local_anchor(TORSO_INITIAL, (3.588733341630704, -4.526434658500262))
KNEE_LEFT = local_anchor(LEFT_CALF_INITIAL, (3.384323411985692, 3.5168931240916876))
KNEE_RIGHT = local_anchor(RIGHT_CALF_INITIAL, (1.4982369235492752, 4.175600306005656))


@dataclass(frozen=True)
class GaitConfig:
    # Coefficient zero is the exact native reward control, with identical logging.
    coefficient: float = 0.0
    pelvis_low_m: float = 0.30
    pelvis_target_m: float = 0.65
    knee_clearance_m: float = 0.12
    max_tilt_degrees: float = 45.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Gait settings must be finite")
        if not 0 <= self.coefficient <= 1:
            raise ValueError("Gait coefficient must be in [0, 1]")
        if not 0 <= self.pelvis_low_m < self.pelvis_target_m:
            raise ValueError("Pelvis thresholds must be ordered")
        if self.knee_clearance_m <= 0 or not 0 < self.max_tilt_degrees < 90:
            raise ValueError("Invalid clearance or tilt threshold")

    def identity(self):
        return {"id": "gait-v1", **asdict(self)}

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.pop("id", "gait-v1") != "gait-v1":
            raise ValueError("Unknown gait definition")
        return cls(**value)


def world_anchor(body, local):
    c, s = math.cos(body[2]), math.sin(body[2])
    return np.array((body[0] + c * local[0] - s * local[1], body[1] + s * local[0] + c * local[1]))


def measure(pose, config):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (12, 3) or not np.isfinite(pose).all():
        raise ValueError("Expected finite raw positions/angles for all 12 bodies")
    pelvis = (world_anchor(pose[0], HIP_LEFT) + world_anchor(pose[0], HIP_RIGHT)) / 2
    neck = world_anchor(pose[0], NECK)
    knees = [world_anchor(pose[3], KNEE_LEFT), world_anchor(pose[8], KNEE_RIGHT)]
    pelvis_height = float((GROUND_Y - pelvis[1]) / 10)
    clearance = [float((GROUND_Y - knee[1]) / 10) for knee in knees]
    tilt = abs(math.degrees(math.atan2(neck[0] - pelvis[0], pelvis[1] - neck[1])))
    height_score = np.clip(
        (pelvis_height - config.pelvis_low_m) / (config.pelvis_target_m - config.pelvis_low_m), 0, 1
    )
    knee_score = np.clip(min(clearance) / config.knee_clearance_m, 0, 1)
    # Full credit near vertical, smoothly falling to zero at 90 degrees.
    tilt_score = np.clip((90 - tilt) / (90 - config.max_tilt_degrees), 0, 1)
    return {
        "pelvis_height_m": pelvis_height,
        "torso_tilt_degrees": tilt,
        "left_knee_clearance_m": clearance[0],
        "right_knee_clearance_m": clearance[1],
        "knee_near_ground": min(clearance) < config.knee_clearance_m,
        "upright": bool(
            pelvis_height >= config.pelvis_target_m
            and min(clearance) >= config.knee_clearance_m
            and tilt <= config.max_tilt_degrees
        ),
        "posture_quality": float(height_score * knee_score * tilt_score),
    }


def reward_adjustment(positive_speed_reward, quality, coefficient):
    # Native forward-speed reward ranges from 0.1x (poor posture) to 2x (upright).
    # No standing bonus: without forward motion, the additional reward is zero.
    return coefficient * max(0.0, positive_speed_reward) * (1.9 * quality - 0.9)


class GaitAccumulator:
    def __init__(self):
        self.steps = 0
        self.sums = dict.fromkeys(
            (
                "pelvis_height_m",
                "torso_tilt_degrees",
                "posture_quality",
                "upright",
                "knee_near_ground",
            ),
            0.0,
        )
        self.forward_m = self.upright_forward_m = 0.0
        self.native_return = self.adjustment_return = 0.0

    def add(self, metrics, delta_m, native_reward, adjustment):
        self.steps += 1
        for key in self.sums:
            self.sums[key] += metrics[key]
        forward = max(0.0, delta_m)
        self.forward_m += forward
        self.upright_forward_m += forward * metrics["upright"]
        self.native_return += float(native_reward)
        self.adjustment_return += adjustment

    def summary(self):
        return {
            "steps": self.steps,
            **{f"mean_{k}": v / self.steps if self.steps else None for k, v in self.sums.items()},
            "forward_distance_m": self.forward_m,
            "upright_forward_distance_m": self.upright_forward_m,
            "upright_forward_fraction": self.upright_forward_m / self.forward_m
            if self.forward_m
            else 0.0,
            "native_return": self.native_return,
            "gait_adjustment_return": self.adjustment_return,
        }


class GaitWrapper(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config

    def _info(self, info):
        raw = self.env.unwrapped.raw_observation
        pose = np.asarray(raw).reshape(12, 5)[:, :3]
        return {**info, "gait_pose": pose.tolist(), "gait": measure(pose, self.config)}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.accumulator = GaitAccumulator()
        self.previous_distance = float(info["distance"])
        return obs, self._info(info)

    def step(self, action):
        # The native engine starts reward calculation at its zero-valued reaction,
        # even after reset. Match that convention exactly in the speed component.
        previous = self.env.unwrapped.last_reaction
        obs, native, terminated, truncated, info = self.env.step(action)
        info = self._info(info)
        dt = float(info["time"] - previous.time)
        if dt <= 0:
            raise ValueError("Game time did not advance")
        speed_reward = float((info["distance"] - previous.distance) / dt) * 0.01
        adjustment = reward_adjustment(
            speed_reward, info["gait"]["posture_quality"], self.config.coefficient
        )
        self.accumulator.add(
            info["gait"], float(info["distance"]) - self.previous_distance, native, adjustment
        )
        self.previous_distance = float(info["distance"])
        info["native_reward"] = float(native)
        info["gait_adjustment"] = adjustment
        if terminated or truncated:
            info["gait_episode"] = self.accumulator.summary()
        # Preserve the exact reward object and precision for the control.
        return (
            obs,
            native if self.config.coefficient == 0 else float(native) + adjustment,
            terminated,
            truncated,
            info,
        )
