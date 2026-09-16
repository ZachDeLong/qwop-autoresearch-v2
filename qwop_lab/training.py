"""PPO training with explicit initialization, architecture, and step checkpoints."""

import json
import random
import time
import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .agents import configure_torch, import_legacy_weights
from .architectures import Architecture
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

# A fresh learner needs a training rate appropriate for exploration, unlike the
# conservative historical-weight pilot. Frozen across all architecture arms.
FRESH_PPO_CONFIG = {
    **PPO_CONFIG,
    "n_steps": 2048,
    "n_epochs": 6,
    "learning_rate": 3e-4,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}


def save_checkpoint(model, path, architecture):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial.zip")
    model.save(partial)
    os.replace(partial, path)
    write_json(
        path.with_suffix(".json"),
        {
            "sha256": sha256(path),
            "algorithm": "PPO",
            "architecture": architecture.identity(),
            "trained_steps": model.num_timesteps,
        },
    )


class Recorder(BaseCallback):
    def __init__(self, path, architecture=None, checkpoint_interval=None, max_seconds=None):
        super().__init__()
        self.path = path
        self.start = time.perf_counter()
        self.architecture = architecture or Architecture()
        self.checkpoint_interval = checkpoint_interval
        self.max_seconds = max_seconds
        self.timed_out = False

    def _on_rollout_start(self):
        if self.num_timesteps:
            optimizer = {
                key: float(value)
                for key, value in self.logger.name_to_value.items()
                if key.startswith("train/") and isinstance(value, (int, float))
            }
            with (self.path.parent / "optimization.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"steps": self.num_timesteps, **optimizer}) + "\n")
        # SB3 calls this after the previous rollout's gradient updates finish.
        if (
            self.checkpoint_interval
            and self.num_timesteps
            and self.num_timesteps % self.checkpoint_interval == 0
        ):
            self._save_progress_checkpoint()

    def _on_training_end(self):
        if self.checkpoint_interval and not self.timed_out:
            self._save_progress_checkpoint()

    def _save_progress_checkpoint(self):
        save_checkpoint(
            self.model,
            self.path.parent / "checkpoints" / f"step-{self.num_timesteps:09d}.zip",
            self.architecture,
        )

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
            write_json(
                self.path.parent / "progress.json",
                {
                    "steps": self.num_timesteps,
                    "wall_seconds": elapsed,
                    "steps_per_second": self.num_timesteps / elapsed,
                },
            )
            print(
                f"training: {self.num_timesteps:,} steps, "
                f"{self.num_timesteps / elapsed:.0f} steps/s",
                flush=True,
            )
        self.timed_out = bool(
            self.max_seconds and time.perf_counter() - self.start >= self.max_seconds
        )
        return not self.timed_out


def train(
    checkpoint,
    out_dir,
    ledger,
    steps=131072,
    time_cost=10,
    seed=42,
    *,
    architecture=None,
    config=None,
    checkpoint_interval=None,
    max_parameters=250000,
    max_seconds=None,
):
    architecture = architecture or Architecture()
    config = dict(PPO_CONFIG if config is None else config)
    if steps <= 0 or steps % config["n_steps"]:
        raise ValueError("Training steps must be a positive multiple of the rollout length")
    if checkpoint_interval and (
        checkpoint_interval <= 0
        or checkpoint_interval % config["n_steps"]
        or steps % checkpoint_interval
    ):
        raise ValueError("Checkpoints must divide the budget and align with PPO rollouts")
    if checkpoint is not None and architecture != Architecture():
        raise ValueError("Historical weights require the original 128x128 Tanh architecture")
    checkpoint = Path(checkpoint).resolve() if checkpoint is not None else None
    checkpoint_hash = sha256(checkpoint) if checkpoint else None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    configure_torch(seed)
    random.seed(seed)
    meta = {
        "status": "running",
        "algorithm": "stable-baselines3 PPO",
        "source_checkpoint": str(checkpoint) if checkpoint else None,
        "source_checkpoint_sha256": checkpoint_hash,
        "initialization": "legacy_weights" if checkpoint else "fresh_seeded_weights",
        "seed": seed,
        "requested_steps": steps,
        "training_reward": {"time_cost_mult": time_cost, "success_reward": 50},
        "ppo_config": config,
        "architecture": architecture.identity(),
        "checkpoint_interval": checkpoint_interval,
        "max_parameters": max_parameters,
        "max_training_seconds": max_seconds,
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
            **config,
            policy_kwargs=architecture.policy_kwargs(),
        )
        meta["trainable_parameters"] = sum(
            p.numel() for p in model.policy.parameters() if p.requires_grad
        )
        if meta["trainable_parameters"] > max_parameters:
            raise ValueError("Architecture exceeds the campaign parameter cap")
        if checkpoint is not None:
            import_legacy_weights(model, checkpoint)
        save_checkpoint(model, out_dir / "initial.zip", architecture)
        recorder = Recorder(
            out_dir / "episodes.jsonl", architecture, checkpoint_interval, max_seconds
        )
        model.learn(total_timesteps=steps, callback=recorder)
        if recorder.timed_out:
            raise TimeoutError("Trial reached its training wall-time cap")
        if lease.used != steps:
            raise RuntimeError(f"Unexpected step count: {lease.used} != {steps}")
        save_checkpoint(model, out_dir / "final.zip", architecture)
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
