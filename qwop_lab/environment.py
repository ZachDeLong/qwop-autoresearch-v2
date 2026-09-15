"""Versioned physics contract and reproducible episode starts."""

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import qwop_gym.envs.v1.qwop_env as engine

from .artifacts import ROOT, digest, read_json, sha256
from .browser import BROWSER_FLAGS, LabServer, StrictClient
from .budget import StepMeter

PROTOCOL = {
    "id": "qwop-lab-v1",
    "frames_per_step": 4,
    "max_episode_steps": 5000,
    "observation": "qwop-gym-1.0.1-normalized-60-float32",
    "actions": "qwop-gym-1.0.1-full-16-enumerated-key-combinations",
    "completion": "qwop-gym native is_success at terminal; includes 105m fallback",
    "score_time": "info.time * 10; raw info.time retained separately",
    "first_100m": "first observed distance >=100; sampled every four physics frames",
    "reset": "hard reset(seed), followed by case.soft_resets soft resets",
    "inference": "deterministic argmax",
    "browser_flags": BROWSER_FLAGS,
}


@dataclass(frozen=True)
class Case:
    seed: int
    soft_resets: int = 0

    def __post_init__(self):
        if not 1 <= self.seed < 2**31 or not 0 <= self.soft_resets <= 10:
            raise ValueError("Use seed 1..2^31-1 and soft_resets 0..10")

    @property
    def id(self):
        return f"seed-{self.seed}-soft-{self.soft_resets}"


VALIDATION_CASES = [Case(seed, phase) for seed in (101, 202, 303) for phase in (0, 1)]


def runtime():
    manifest = read_json(ROOT / ".runtime" / "runtime.json")
    for item in manifest["files"].values():
        if not Path(item["path"]).is_file() or sha256(item["path"]) != item["sha256"]:
            raise RuntimeError(f"Runtime file changed or missing: {item['path']}. Re-bootstrap.")
    return manifest


def contract(manifest):
    value = {**PROTOCOL, "runtime_id": manifest["runtime_id"]}
    return {**value, "contract_id": digest(value)}


def make_env(lease=None, time_cost_mult=10, success_reward=50):
    manifest = runtime()
    engine.WSServer = LabServer
    engine.WSClient = StrictClient
    os.environ["QWOP_LAB_BROWSER_VERSION"] = manifest["browser_version"]
    env = engine.QwopEnv(
        browser=manifest["files"]["browser"]["path"],
        driver=manifest["files"]["driver"]["path"],
        seed=1,
        frames_per_step=4,
        reload_on_reset=False,
        render_mode="rgb_array",
        auto_draw=False,
        game_in_browser=True,
        time_cost_mult=time_cost_mult,
        success_reward=success_reward,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=5000)
    return StepMeter(env, lease) if lease else env


def reset_case(env, case):
    obs, info = env.reset(seed=case.seed)
    for _ in range(case.soft_resets):
        obs, info = env.reset()
    if obs.shape != (60,) or not np.isfinite(obs).all():
        raise RuntimeError("Invalid initial observation")
    return obs, info


def case_dict(case):
    return {"id": case.id, **asdict(case)}
