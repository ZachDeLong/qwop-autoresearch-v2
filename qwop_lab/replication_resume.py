"""Explicit budget-preserving continuation of a logged replication interruption.

This is not exact mid-episode resume: partial rollout/game state is discarded and
the action RNG is reseeded. The failed attempt remains immutable. Reports also
show the unaffected seed pairs separately.
"""

import argparse
import multiprocessing
from pathlib import Path
import time

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from .agents import configure_torch
from .architectures import Architecture
from . import training
from .artifacts import ROOT, read_json, sha256, write_json as write_json_once
from .budget import Ledger
from .environment import Case, contract, make_env, runtime
from .evaluation import evaluate
from .replay import replay
from .replication import aggregate, now
from .training import Recorder, save_checkpoint


def write_json(path, value):
    """Retry a locked progress-file replacement without advancing the game."""
    for attempt in range(10):
        try:
            return write_json_once(path, value)
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(0.02 * 2**attempt, 0.2))


def verify(folder):
    folder = Path(folder)
    protocol = read_json(folder / "protocol.json")
    assert sha256(folder / "protocol.json") == (folder / "protocol.sha256").read_text().strip()
    amendment = read_json(folder / "resume-amendment.json")
    assert (
        sha256(folder / "resume-amendment.json")
        == (folder / "resume-amendment.sha256").read_text().strip()
    )
    assert amendment["protocol_sha256"] == sha256(folder / "protocol.json")
    assert amendment["implementation_sha256"] == sha256(__file__)
    for relative, expected in protocol["protected_sources"].items():
        allowed = amendment["source_changes"].get(relative, expected)
        assert sha256(ROOT / relative) == allowed, f"Unapproved source change: {relative}"
    assert contract(runtime()) == protocol["contract"]
    path = folder / amendment["design"] / f"seed-{amendment['seed']}"
    assert sha256(path / "result.json") == amendment["failed_result_sha256"]
    assert sha256(path / "training/manifest.json") == amendment["failed_manifest_sha256"]
    assert sha256(path / "training/interrupted.zip") == amendment["interrupted_checkpoint_sha256"]
    return protocol, amendment


def amend(folder):
    folder = Path(folder).resolve()
    if (folder / "resume-amendment.json").exists():
        raise FileExistsError("Amendment already frozen")
    protocol = read_json(folder / "protocol.json")
    assert sha256(folder / "protocol.json") == (folder / "protocol.sha256").read_text().strip()
    design, seed = "baseline", 29
    path = folder / design / f"seed-{seed}"
    failed = read_json(path / "result.json")
    meta = read_json(path / "training/manifest.json")
    assert failed["status"] == meta["status"] == "failed"
    assert "PermissionError" in failed["error"] and "progress.json" in failed["error"]
    spent = Ledger(path / "training.sqlite").status()["charged_steps"]
    assert spent == meta["actual_steps"]
    model = PPO.load(path / "training/interrupted.zip", device="cpu")
    assert model.num_timesteps == spent
    remaining = protocol["training_steps_per_run"] - spent
    assert remaining > 0 and remaining % protocol["ppo_config"]["n_steps"] == 0
    changes = {
        relative: sha256(ROOT / relative)
        for relative, original in protocol["protected_sources"].items()
        if sha256(ROOT / relative) != original
    }
    assert set(changes) <= {"qwop_lab/artifacts.py"}
    for other_seed in protocol["seeds"]:
        for other_design in protocol["designs"]:
            if (other_design, other_seed) != (design, seed):
                other_result = folder / other_design / f"seed-{other_seed}/result.json"
                if other_result.exists():
                    assert read_json(other_result)["status"] != "failed"
    value = {
        "id": "replication-log-lock-continuation-v1",
        "created_utc": now(),
        "protocol_sha256": sha256(folder / "protocol.json"),
        "implementation_sha256": sha256(__file__),
        "source_changes": changes,
        "design": design,
        "seed": seed,
        "failed_result_sha256": sha256(path / "result.json"),
        "failed_manifest_sha256": sha256(path / "training/manifest.json"),
        "interrupted_checkpoint_sha256": sha256(path / "training/interrupted.zip"),
        "spent_training_steps": spent,
        "remaining_training_steps": remaining,
        "optimizer_updates_at_interrupt": model._n_updates,
        "discarded_rollout_interactions": spent
        - model._n_updates // model.n_epochs * model.n_steps,
        "changes": [
            "Allow one continuation from saved weights and optimizer, within the existing ledger cap",
            "Preserve the original failed result and manifest; write a separate recovered result",
            "Reset the game and action RNG to the original training seed; this is not exact resume",
            "Discard unoptimized rollout data; account for those interactions as already spent",
            "Retry transient Windows file-replacement permission errors without taking extra game actions",
            "The continuation process binds the frozen Recorder's JSON writer to this amendment's bounded-retry writer; original training processes keep their frozen source",
            "Report all three seed pairs and a separate sensitivity analysis of the two unaffected pairs",
            "No extra training steps or new random initialization are allowed",
        ],
    }
    write_json(folder / "resume-amendment.json", value)
    (folder / "resume-amendment.sha256").write_text(sha256(folder / "resume-amendment.json") + "\n")
    return value


def resume(folder):
    folder = Path(folder).resolve()
    protocol, amendment = verify(folder)
    design, seed = amendment["design"], amendment["seed"]
    path = folder / design / f"seed-{seed}"
    assert len(list(folder.glob("*/seed-*/running.lock"))) < protocol["max_workers"]
    out = path / "recovery-training"
    out.mkdir(exist_ok=False)
    ledger = Ledger(path / "training.sqlite")
    assert ledger.status()["remaining_steps"] == amendment["remaining_training_steps"]
    lease = ledger.lease("recorded-continuation:baseline/seed-29")
    arch = Architecture.from_dict(protocol["designs"][design])
    configure_torch(seed)
    # This dedicated continuation process uses the hash-recorded writer above.
    training.write_json = write_json
    env = model = None
    meta = {
        "status": "running",
        "seed": seed,
        "initialization": "recorded_interrupted_continuation",
        "ppo_config": protocol["ppo_config"],
        "contract": protocol["contract"],
        "architecture": arch.identity(),
        "amendment_sha256": sha256(folder / "resume-amendment.json"),
        "requested_steps": amendment["remaining_training_steps"],
        "prior_spent_steps": amendment["spent_training_steps"],
        "source_checkpoint_sha256": amendment["interrupted_checkpoint_sha256"],
    }
    write_json(out / "manifest.json", meta)
    start = time.perf_counter()
    try:
        env = Monitor(make_env(lease))
        model = PPO.load(path / "training/interrupted.zip", env=env, device="cpu")
        model.set_random_seed(seed)
        assert model.num_timesteps == amendment["spent_training_steps"]
        meta["trainable_parameters"] = sum(
            p.numel() for p in model.policy.parameters() if p.requires_grad
        )
        save_checkpoint(model, out / "initial.zip", arch)
        recorder = Recorder(
            out / "episodes.jsonl",
            arch,
            protocol["checkpoint_interval"],
            protocol["training_wall_seconds_per_run"]
            - read_json(path / "training/manifest.json")["wall_seconds"],
        )
        model.learn(
            total_timesteps=amendment["remaining_training_steps"],
            reset_num_timesteps=False,
            callback=recorder,
        )
        assert not recorder.timed_out
        assert lease.used == amendment["remaining_training_steps"]
        assert model.num_timesteps == protocol["training_steps_per_run"]
        save_checkpoint(model, out / "final.zip", arch)
        meta.update(
            status="complete",
            final_checkpoint_sha256=sha256(out / "final.zip"),
            optimizer_updates=model._n_updates,
        )
    except BaseException as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if model:
            model.save(out / "interrupted.zip")
        raise
    finally:
        try:
            if env:
                env.close()
        finally:
            lease.close()
            meta.update(
                actual_steps=lease.used,
                cumulative_steps=lease.used + amendment["spent_training_steps"],
                wall_seconds=time.perf_counter() - start,
                budget=ledger.status(),
            )
            write_json(out / "manifest.json", meta)
    return evaluate_recovered(folder)


def evaluate_recovered(folder):
    folder = Path(folder).resolve()
    protocol, amendment = verify(folder)
    path = folder / amendment["design"] / f"seed-{amendment['seed']}"
    meta = read_json(path / "recovery-training/manifest.json")
    assert meta["status"] == "complete"
    ledger = Ledger(path / "evaluation.sqlite")
    cases = [Case(**c) for c in protocol["evaluation_cases"]]
    result = {
        "status": "running",
        "design": amendment["design"],
        "seed": amendment["seed"],
        "protocol_sha256": sha256(folder / "protocol.json"),
        "amendment_sha256": sha256(folder / "resume-amendment.json"),
        "checkpoints": [],
    }
    for step in range(
        protocol["checkpoint_interval"],
        protocol["training_steps_per_run"] + 1,
        protocol["checkpoint_interval"],
    ):
        model = path / "recovery-training/checkpoints" / f"step-{step:09d}.zip"
        if step == protocol["training_steps_per_run"]:
            model = path / "recovery-training/final.zip"
        output = path / f"eval-{step:09d}"
        if output.exists():
            manifest = read_json(output / "manifest.json")
            assert manifest["status"] == "complete" and manifest["model"]["sha256"] == sha256(model)
            summary = read_json(output / "summary.json")
        else:
            summary = evaluate(model, cases, output, ledger, f"baseline/seed-29-resumed@{step}")
        result["checkpoints"].append(
            {"steps": step, "summary": summary, "checkpoint_sha256": sha256(model)}
        )
    video = path / "replay.mp4"
    if video.exists():
        video_meta = read_json(video.with_suffix(".json"))
        assert video_meta["verified"] and video_meta["video_sha256"] == sha256(video)
    else:
        video_meta = replay(
            path / f"eval-{protocol['training_steps_per_run']:09d}/seed-101-soft-0.json",
            video,
            ledger,
            "baseline | seed 29 | recorded continuation | 1.05M",
        )
    result.update(
        status="complete",
        final_summary=result["checkpoints"][-1]["summary"],
        trainable_parameters=meta["trainable_parameters"],
        final_checkpoint_sha256=meta["final_checkpoint_sha256"],
        video=video_meta,
    )
    write_json(path / "recovered-result.json", result)
    return result


def finish(folder):
    folder = Path(folder).resolve()
    protocol, amendment = verify(folder)
    if (folder / "comparison.json").exists():
        raise FileExistsError("Already finalized")
    results = {}
    training = evaluation = 0
    wall = 0.0
    for seed in protocol["seeds"]:
        for design in protocol["designs"]:
            path = folder / design / f"seed-{seed}"
            recovered = (design, seed) == (amendment["design"], amendment["seed"])
            result = read_json(path / ("recovered-result.json" if recovered else "result.json"))
            assert result["status"] == "complete"
            trained = Ledger(path / "training.sqlite").status()
            assert (
                trained["charged_steps"]
                == trained["confirmed_steps"]
                == protocol["training_steps_per_run"]
            )
            results[(design, seed)] = result
            training += trained["charged_steps"]
            evaluation += Ledger(path / "evaluation.sqlite").status()["charged_steps"]
            wall += read_json(path / "training/manifest.json")["wall_seconds"]
            if recovered:
                wall += read_json(path / "recovery-training/manifest.json")["wall_seconds"]
    clean_seeds = [seed for seed in protocol["seeds"] if seed != amendment["seed"]]
    comparison = {
        "status": "complete_with_recorded_continuation",
        "created_utc": now(),
        "protocol": protocol,
        "amendment": amendment,
        "summary": aggregate(results, protocol["seeds"]),
        "uninterrupted_seed_sensitivity": aggregate(results, clean_seeds),
        "costs": {
            "training_steps": training,
            "evaluation_and_replay_steps": evaluation,
            "total_steps": training + evaluation,
            "summed_training_process_seconds": wall,
            "discarded_rollout_interactions": amendment["discarded_rollout_interactions"],
        },
    }
    write_json(folder / "comparison.json", comparison)
    return comparison


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["amend", "resume", "evaluate", "finish"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(
        {"amend": amend, "resume": resume, "evaluate": evaluate_recovered, "finish": finish}[
            args.action
        ](args.out),
        flush=True,
    )
