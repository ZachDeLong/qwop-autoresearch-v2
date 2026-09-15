"""One maintained PPO trainer; reward is the only pilot treatment variable."""

import json
import random
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .agents import configure_torch, import_legacy_weights
from .artifacts import provenance, sha256, write_json
from .environment import contract, make_env, runtime

PPO_CONFIG = {
    "n_steps": 512,
    "batch_size": 128,
    "n_epochs": 4,
    "learning_rate": 3e-5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.1,
    "ent_coef": 0.015,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "device": "cpu",
    "verbose": 0,
}


class Recorder(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.start = time.perf_counter()

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                row = {
                    "step": self.num_timesteps,
                    "return": info["episode"]["r"],
                    "length": info["episode"]["l"],
                    "distance": float(info["distance"]),
                    "score_time": float(info["time"]) * 10,
                    "is_success": bool(info["is_success"]),
                }
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        if self.num_timesteps % 8192 == 0:
            elapsed = time.perf_counter() - self.start
            print(
                f"training: {self.num_timesteps:,} steps, "
                f"{self.num_timesteps / elapsed:.0f} steps/s",
                flush=True,
            )
        return True


def train(checkpoint, out_dir, ledger, steps=131072, time_cost=10, seed=42):
    if steps <= 0 or steps % PPO_CONFIG["n_steps"]:
        raise ValueError("Training steps must be a positive multiple of rollout length 512")
    checkpoint = Path(checkpoint).resolve()
    checkpoint_hash = sha256(checkpoint)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    configure_torch(seed)
    random.seed(seed)
    meta = {
        "status": "running",
        "algorithm": "stable-baselines3 PPO",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": checkpoint_hash,
        "seed": seed,
        "requested_steps": steps,
        "training_reward": {"time_cost_mult": time_cost, "success_reward": 50},
        "ppo_config": PPO_CONFIG,
        "architecture": {"pi": [128, 128], "vf": [128, 128], "activation": "Tanh"},
        "contract": contract(runtime()),
        "provenance": provenance(),
        "selection": "Final checkpoint at fixed budget; no validation-based selection",
        "reset": "Initial reset(seed=training_seed), then native soft resets",
    }
    write_json(out_dir / "manifest.json", meta)
    lease = ledger.lease(f"train:{out_dir.name}")
    env = model = None
    start = time.perf_counter()
    try:
        env = Monitor(make_env(lease, time_cost_mult=time_cost, success_reward=50))
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            **PPO_CONFIG,
            policy_kwargs={
                "net_arch": dict(pi=[128, 128], vf=[128, 128]),
                "activation_fn": torch.nn.Tanh,
            },
        )
        import_legacy_weights(model, checkpoint)
        model.save(out_dir / "initial.zip")
        model.learn(total_timesteps=steps, callback=Recorder(out_dir / "episodes.jsonl"))
        if lease.used != steps:
            raise RuntimeError(f"Unexpected step count: {lease.used} != {steps}")
        model.save(out_dir / "final.zip")
        meta.update(status="complete", final_checkpoint_sha256=sha256(out_dir / "final.zip"))
        return out_dir / "final.zip"
    except BaseException as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if model is not None:
            model.save(out_dir / "interrupted.zip")
        raise
    finally:
        try:
            if env:
                env.close()
        finally:
            lease.close()
            meta.update(
                actual_steps=lease.used,
                wall_seconds=time.perf_counter() - start,
                budget=ledger.status(),
            )
            write_json(out_dir / "manifest.json", meta)
