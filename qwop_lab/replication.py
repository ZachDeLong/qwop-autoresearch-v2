"""Frozen, fresh-seed replication of two previously selected PPO designs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from .architectures import Architecture
from .artifacts import ROOT, read_json, sha256, write_json
from .budget import Ledger
from .environment import Case, contract, runtime
from .evaluation import SCORING, evaluate
from .replay import replay
from .training import FRESH_PPO_CONFIG, train

SEEDS = (17, 29, 61)
DESIGNS = {
    "baseline": {"layers": [128, 128], "activation": "Tanh"},
    "kinematic": {
        "layers": [64, 64],
        "activation": "Tanh",
        "extractor": "qwop_lab.candidates.kinematic_v2:KinematicExtractor",
    },
}


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize(folder, previous=None):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    sources = {
        str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
        for p in sorted((ROOT / "qwop_lab").rglob("*.py"))
    }
    protocol = {
        "id": "architecture-replication-v1",
        "created_utc": now(),
        "seeds": list(SEEDS),
        "designs": DESIGNS,
        "architecture_identities": {
            name: Architecture.from_dict(spec).identity() for name, spec in DESIGNS.items()
        },
        "training_steps_per_run": 1048576,
        "checkpoint_interval": 262144,
        "evaluation_and_replay_cap_per_run": 50000,
        "training_wall_seconds_per_run": 7200,
        "max_parameters": 250000,
        "max_workers": 3,
        "ppo_config": FRESH_PPO_CONFIG,
        "training_reward": {"time_cost_mult": 10, "success_reward": 50},
        "initialization": "Fresh seeded weights for every run; no old weights or continuation",
        "evaluation_cases": [{"seed": 101, "soft_resets": phase} for phase in (0, 1)],
        "replay_case": {"seed": 101, "soft_resets": 0},
        "contract": contract(runtime()),
        "scoring": SCORING,
        "analysis": {
            "unit": "Independent training run; three seed pairs, not six independent reset cases",
            "reliability": "Show both-phase finishing runs, all phase completions, and every seed",
            "speed": "Per-seed mean 100m game seconds across both phases when both finish; "
            "paired percent changes only when both designs finish both phases",
            "secondary": "Best valid 100m time and conditional finish time retained per run",
            "learning": "Evaluate fixed quarter-budget checkpoints; no checkpoint selection",
            "selection": "Every final checkpoint is reported; no seed or model selection",
            "interpretation": "Descriptive small-sample replication of a selected architecture; "
            "not a replicated AI research campaign or a held-out robustness evaluation",
            "gait": "Inspect fixed-case final replays; speed alone does not establish running",
        },
        "failure_policy": "Preserve failed attempts and all charges. No automatic training "
        "restarts or extra training budget. Successful training may proceed to its remaining "
        "evaluation stages on a later invocation; existing artifacts are verified and retained. "
        "Any failed training or evaluation makes the batch incomplete and must be reported.",
        "protected_sources": sources,
        "prior_evidence": {
            "path": "results/architecture-001/report.md",
            "sha256": sha256(ROOT / "results/architecture-001/report.md"),
        },
        "paid_model_api_calls": 0,
    }
    if previous is not None:
        previous = Path(previous).resolve()
        previous_protocol = read_json(previous / "protocol.json")
        failures = []
        for seed in SEEDS:
            for design in DESIGNS:
                path = previous / design / f"seed-{seed}"
                result = read_json(path / "result.json")
                if result["status"] != "failed":
                    raise ValueError(
                        "Startup replacement requires all previous runs to have failed"
                    )
                for kind in ("training", "evaluation"):
                    if Ledger(path / f"{kind}.sqlite").status()["charged_steps"] != 0:
                        raise ValueError("Startup replacement cannot discard any spent game steps")
                failures.append({"path": str(path), "result_sha256": sha256(path / "result.json")})
        for key in ("seeds", "designs", "ppo_config", "training_steps_per_run", "contract"):
            if protocol[key] != previous_protocol[key]:
                raise ValueError(f"Startup replacement changed experimental settings: {key}")
        protocol["startup_recovery"] = {
            "previous_protocol_sha256": sha256(previous / "protocol.json"),
            "previous_folder": str(previous),
            "failed_runs": failures,
            "previous_charged_steps": 0,
            "reason": "All six launches failed before training. Clearing orphaned headless "
            "browsers restored startup. Cleanup and registration timeout handling were fixed "
            "before this replacement protocol was frozen. Physics and learning settings match.",
        }
    write_json(folder / "protocol.json", protocol)
    (folder / "protocol.sha256").write_text(sha256(folder / "protocol.json") + "\n")
    for relative in sources:
        dest = folder / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
    for seed in SEEDS:
        for design in DESIGNS:
            path = folder / design / f"seed-{seed}"
            Ledger(path / "training.sqlite", protocol["training_steps_per_run"])
            Ledger(path / "evaluation.sqlite", protocol["evaluation_and_replay_cap_per_run"])
    return protocol


def verify_protocol(folder):
    folder = Path(folder)
    if sha256(folder / "protocol.json") != (folder / "protocol.sha256").read_text().strip():
        raise ValueError("Replication protocol changed")
    protocol = read_json(folder / "protocol.json")
    for relative, expected in protocol["protected_sources"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Protected source changed: {relative}")
    if contract(runtime()) != protocol["contract"]:
        raise ValueError("Replication runtime changed")
    return protocol


def run_trial(folder, design, seed):
    folder = Path(folder).resolve()
    protocol = verify_protocol(folder)
    if design not in protocol["designs"] or seed not in protocol["seeds"]:
        raise ValueError("Run is not in the frozen replication protocol")
    path = folder / design / f"seed-{seed}"
    # Exclusive claim prevents two invocations from spending a run's allowance.
    with (path / "running.lock").open("x") as f:
        f.write(now())
    training_ledger = Ledger(path / "training.sqlite")
    evaluation_ledger = Ledger(path / "evaluation.sqlite")
    result = {
        "status": "running",
        "design": design,
        "seed": seed,
        "started_utc": now(),
        "protocol_sha256": sha256(folder / "protocol.json"),
        "checkpoints": [],
    }
    already_complete = False
    try:
        if (path / "result.json").exists():
            prior = read_json(path / "result.json")
            if prior["status"] == "complete":
                already_complete = True
                return prior
            result["prior_invocation"] = prior
        write_json(path / "result.json", result)
        steps = protocol["training_steps_per_run"]
        interval = protocol["checkpoint_interval"]
        model = path / "training/final.zip"
        if (path / "training").exists():
            meta = read_json(path / "training/manifest.json")
            if meta["status"] != "complete" or meta["actual_steps"] != steps:
                raise ValueError("Incomplete training retained; automatic restart is forbidden")
            if sha256(model) != meta["final_checkpoint_sha256"]:
                raise ValueError("Completed model changed")
        else:
            model = train(
                None,
                path / "training",
                training_ledger,
                steps=steps,
                seed=seed,
                architecture=Architecture.from_dict(protocol["designs"][design]),
                config=protocol["ppo_config"],
                checkpoint_interval=interval,
                max_parameters=protocol["max_parameters"],
                max_seconds=protocol["training_wall_seconds_per_run"],
            )
        verify_protocol(folder)
        cases = [Case(**c) for c in protocol["evaluation_cases"]]
        for step in range(interval, steps + 1, interval):
            checkpoint = (
                model if step == steps else path / "training/checkpoints" / f"step-{step:09d}.zip"
            )
            evaluation = path / f"eval-{step:09d}"
            if evaluation.exists():
                manifest = read_json(evaluation / "manifest.json")
                if manifest["status"] != "complete":
                    raise ValueError("Failed evaluation retained; no silent retry")
                if manifest["model"]["sha256"] != sha256(checkpoint):
                    raise ValueError("Saved evaluation belongs to different weights")
                summary = read_json(evaluation / "summary.json")
            else:
                summary = evaluate(
                    checkpoint, cases, evaluation, evaluation_ledger, f"{design}/seed-{seed}@{step}"
                )
            result["checkpoints"].append(
                {"steps": step, "summary": summary, "checkpoint_sha256": sha256(checkpoint)}
            )
            write_json(path / "result.json", result)
        video = path / "replay.mp4"
        if video.exists():
            video_meta = read_json(video.with_suffix(".json"))
            if not video_meta["verified"] or sha256(video) != video_meta["video_sha256"]:
                raise ValueError("Existing video is not verified")
        else:
            video_meta = replay(
                path / f"eval-{steps:09d}" / "seed-101-soft-0.json",
                video,
                evaluation_ledger,
                f"{design} | training seed {seed} | 1.05M steps",
            )
        meta = read_json(path / "training/manifest.json")
        result.update(
            status="complete",
            final_summary=result["checkpoints"][-1]["summary"],
            trainable_parameters=meta["trainable_parameters"],
            final_checkpoint_sha256=sha256(model),
            video=video_meta,
        )
        return result
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if not already_complete:
            result.update(
                finished_utc=now(),
                training_budget=training_ledger.status(),
                evaluation_budget=evaluation_ledger.status(),
            )
            write_json(path / "result.json", result)
        (path / "running.lock").unlink()


def status(folder):
    folder = Path(folder)
    protocol = read_json(folder / "protocol.json")
    rows = []
    for seed in protocol["seeds"]:
        for design in protocol["designs"]:
            path = folder / design / f"seed-{seed}"
            result = read_json(path / "result.json") if (path / "result.json").exists() else {}
            progress = path / "training/progress.json"
            meta_path = path / "training/manifest.json"
            meta = read_json(meta_path) if meta_path.exists() else {}
            rows.append(
                {
                    "design": design,
                    "seed": seed,
                    "status": result.get("status", "pending"),
                    "training_status": meta.get("status", "pending"),
                    "progress": read_json(progress) if progress.exists() else None,
                    "actual_steps": meta.get("actual_steps"),
                    "final_summary": result.get("final_summary"),
                    "error": result.get("error"),
                    "evaluated_checkpoints": len(result.get("checkpoints", [])),
                }
            )
    return rows


def run_batch(folder):
    folder = Path(folder).resolve()
    protocol = verify_protocol(folder)

    def worker(design, seed):
        path = folder / design / f"seed-{seed}"
        log = path / "worker.log"
        with log.open("a", encoding="utf-8") as output:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "qwop_lab.replication",
                    "trial",
                    "--out",
                    str(folder),
                    "--design",
                    design,
                    "--seed",
                    str(seed),
                ],
                cwd=ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return {"design": design, "seed": seed, "exit_code": completed.returncode}

    outcomes = []
    with ThreadPoolExecutor(max_workers=protocol["max_workers"]) as pool:
        futures = [
            pool.submit(worker, design, seed)
            for seed in protocol["seeds"]
            for design in protocol["designs"]
        ]
        for future in as_completed(futures):
            outcome = future.result()
            outcomes.append(outcome)
            print(json.dumps(outcome), flush=True)
    write_json(folder / "batch.json", {"finished_utc": now(), "outcomes": outcomes})
    if any(outcome["exit_code"] != 0 for outcome in outcomes):
        raise RuntimeError(
            "Replication has failed workers; inspect the preserved batch and run logs"
        )
    return status(folder)


def aggregate(results, seeds):
    """Keep failures visible and use training seeds as the analysis units."""
    pairs = []
    for seed in seeds:
        pair = {"seed": seed}
        for design in DESIGNS:
            row = results[(design, seed)]
            s = row["final_summary"]
            pair[design] = {
                "finishes": s["valid_100m_finishes"],
                "episodes": s["episodes"],
                "best_100m_game_seconds": s["best_100m_game_seconds"],
                "conditional_mean_100m_game_seconds": s["mean_100m_game_seconds"],
                "both_phase_mean_100m_game_seconds": s["mean_100m_game_seconds"]
                if s["valid_100m_finishes"] == s["episodes"]
                else None,
                "mean_terminal_distance": s["mean_distance"],
            }
        a = pair["baseline"]["both_phase_mean_100m_game_seconds"]
        b = pair["kinematic"]["both_phase_mean_100m_game_seconds"]
        pair["kinematic_time_change_percent"] = 100 * (b / a - 1) if a and b else None
        pairs.append(pair)
    summaries = {}
    for design in DESIGNS:
        times = [
            pair[design]["both_phase_mean_100m_game_seconds"]
            for pair in pairs
            if pair[design]["both_phase_mean_100m_game_seconds"] is not None
        ]
        summaries[design] = {
            "training_runs": len(seeds),
            "runs_finishing_both_phases": len(times),
            "phase_finishes": sum(pair[design]["finishes"] for pair in pairs),
            "phase_episodes": sum(pair[design]["episodes"] for pair in pairs),
            "mean_time_among_both_phase_finishing_runs": statistics.mean(times) if times else None,
            "median_time_among_both_phase_finishing_runs": statistics.median(times)
            if times
            else None,
            "times_by_successful_seed": times,
        }
    deltas = [
        p["kinematic_time_change_percent"]
        for p in pairs
        if p["kinematic_time_change_percent"] is not None
    ]
    return {
        "pairs": pairs,
        "designs": summaries,
        "comparable_speed_pairs": len(deltas),
        "mean_paired_time_change_percent": statistics.mean(deltas) if deltas else None,
        "negative_change_means": "Kinematic design is faster",
        "inference": "Three training seeds; descriptive evidence only. Missing finishes are not imputed or discarded from reliability counts.",
    }


def finish(folder):
    folder = Path(folder).resolve()
    protocol = verify_protocol(folder)
    if (folder / "comparison.json").exists():
        raise FileExistsError("Replication already finalized")
    results = {}
    training = evaluation = 0
    wall = 0.0
    for seed in protocol["seeds"]:
        for design in protocol["designs"]:
            path = folder / design / f"seed-{seed}"
            result = read_json(path / "result.json")
            meta = read_json(path / "training/manifest.json")
            if result["status"] != "complete" or meta["status"] != "complete":
                raise ValueError(f"Incomplete replication run: {design}/{seed}")
            if meta["actual_steps"] != protocol["training_steps_per_run"]:
                raise ValueError("Unequal training interaction budget")
            if result["protocol_sha256"] != sha256(folder / "protocol.json"):
                raise ValueError("Run protocol mismatch")
            if sha256(path / "training/final.zip") != result["final_checkpoint_sha256"]:
                raise ValueError("Final model changed")
            results[(design, seed)] = result
            training += Ledger(path / "training.sqlite").status()["charged_steps"]
            evaluation += Ledger(path / "evaluation.sqlite").status()["charged_steps"]
            wall += meta["wall_seconds"]
    comparison = {
        "status": "complete",
        "created_utc": now(),
        "protocol": protocol,
        "summary": aggregate(results, protocol["seeds"]),
        "costs": {
            "training_steps": training,
            "evaluation_and_replay_steps": evaluation,
            "total_steps": training + evaluation,
            "summed_training_process_seconds": wall,
            "note": "Processes overlap; summed duration is not elapsed wall time or CPU-seconds",
        },
    }
    write_json(folder / "comparison.json", comparison)
    return comparison


def main():
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["init", "trial", "batch", "status", "finish"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--design", choices=list(DESIGNS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--previous", help="Preserved batch that failed with zero game steps")
    args = parser.parse_args()
    if args.action == "trial":
        if args.design is None or args.seed is None:
            parser.error("trial requires --design and --seed")
        result = run_trial(args.out, args.design, args.seed)
    elif args.action == "init":
        result = initialize(args.out, args.previous)
    else:
        result = {"batch": run_batch, "status": status, "finish": finish}[args.action](args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
