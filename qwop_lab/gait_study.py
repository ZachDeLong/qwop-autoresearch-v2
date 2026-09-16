"""A frozen, bounded native-reward versus posture-reward PPO screen."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile

from .artifacts import ROOT, digest, read_json, sha256, write_json
from .budget import Ledger
from .environment import Case, contract, runtime
from .evaluation import evaluate, summarize
from .gait import GaitAccumulator, GaitConfig, measure
from .replay import replay, side_by_side
from .training import FRESH_PPO_CONFIG, train

ARMS = {"baseline": GaitConfig(), "gait": GaitConfig(coefficient=1.0)}


def initialize(folder, steps=2097152, interval=524288, seed=83):
    if steps <= 0 or interval <= 0 or steps % interval or interval % FRESH_PPO_CONFIG["n_steps"]:
        raise ValueError("Positive budgets must align with rollouts and checkpoint intervals")
    if not 1 <= seed < 2**31:
        raise ValueError("Invalid training seed")
    local_runtime = contract(runtime())
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    sources = {
        str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "qwop_lab").rglob("*.py"))
    }
    protocol = {
        "id": "gait-screen-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_steps_per_arm": steps,
        "checkpoint_interval": interval,
        "seed": seed,
        "arms": {name: config.identity() for name, config in ARMS.items()},
        "ppo_config": FRESH_PPO_CONFIG,
        "architecture": "Separate 128x128 Tanh actor and critic; fresh paired initialization",
        "native_reward": {"time_cost_mult": 10, "success_reward": 50},
        "reward_change": "Multiply only positive native speed reward by 0.1 + 1.9 * posture_quality. "
        "Posture quality is the product of clipped hip-height, knee-clearance, and torso-tilt scores. "
        "Native time/failure/completion terms are preserved; no standing bonus.",
        "measurements": "Raw body origins/angles reconstruct hip and knee anchors. Y points down; "
        "ten world units per displayed metre. Clearance is a proximity proxy, not contact sensing. "
        "Upright forward distance sums positive torso displacement on qualifying sampled states; "
        "it is not net race progress and does not establish alternating running.",
        "evaluation_cases": [{"seed": 101, "soft_resets": phase} for phase in (0, 1)],
        "evaluation_gait_config": GaitConfig().identity(),
        "evaluation_cap_per_arm": (2 * (steps // interval) + 1) * 5000,
        "training_wall_seconds_per_arm": 7200,
        "selection": "Report both final policies; intermediate checkpoints are diagnostic only",
        "hypothesis": "Forward-motion-gated posture reward increases upright movement and reduces "
        "knee proximity at the same training budget; completion and distance expose any regression.",
        "limitations": "One paired training seed and two familiar reset phases. Descriptive reward "
        "screen; no statistical superiority, robustness, or convincing running claim without video review. "
        "This Mac runtime is distinct from the archived Windows studies.",
        "failure_policy": "Preserve failures and charges. No automatic training retry, continuation, "
        "budget expansion, or reuse of existing output directories.",
        "contract": local_runtime,
        "protected_sources": sources,
        "prior_report_sha256": sha256(ROOT / "results/replication-002/report.md"),
    }
    write_json(folder / "protocol.json", protocol)
    (folder / "protocol.sha256").write_text(sha256(folder / "protocol.json") + "\n")
    for relative in sources:
        target = folder / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (folder / "requirements-macos.txt").write_text(frozen)
    for arm in ARMS:
        Ledger(folder / arm / "training.sqlite", steps)
        Ledger(folder / arm / "evaluation.sqlite", protocol["evaluation_cap_per_arm"])
    return protocol


def verify_protocol(folder):
    folder = Path(folder)
    if sha256(folder / "protocol.json") != (folder / "protocol.sha256").read_text().strip():
        raise ValueError("Gait protocol changed")
    protocol = read_json(folder / "protocol.json")
    for relative, expected in protocol["protected_sources"].items():
        if sha256(ROOT / relative) != expected or sha256(folder / "source" / relative) != expected:
            raise ValueError(f"Protected source changed: {relative}")
    if contract(runtime()) != protocol["contract"]:
        raise ValueError("Gait runtime changed")
    return protocol


def run_trial(folder, arm):
    folder = Path(folder).resolve()
    protocol = verify_protocol(folder)
    if arm not in protocol["arms"]:
        raise ValueError("Unknown gait arm")
    path = folder / arm
    # Permanent exclusive claim: a failed run cannot be silently restarted.
    with (path / "claimed.json").open("x") as f:
        json.dump({"started_utc": datetime.now(timezone.utc).isoformat()}, f)
    train_ledger, eval_ledger = Ledger(path / "training.sqlite"), Ledger(path / "evaluation.sqlite")
    result = {
        "status": "running",
        "arm": arm,
        "checkpoints": [],
        "protocol_sha256": sha256(folder / "protocol.json"),
    }
    write_json(path / "result.json", result)
    try:
        steps, interval = protocol["training_steps_per_arm"], protocol["checkpoint_interval"]
        model = train(
            None,
            path / "training",
            train_ledger,
            steps=steps,
            seed=protocol["seed"],
            config=protocol["ppo_config"],
            checkpoint_interval=interval,
            max_seconds=protocol["training_wall_seconds_per_arm"],
            gait_config=GaitConfig.from_dict(protocol["arms"][arm]),
        )
        verify_protocol(folder)
        cases = [Case(**c) for c in protocol["evaluation_cases"]]
        for step in range(interval, steps + 1, interval):
            checkpoint = (
                model if step == steps else path / "training/checkpoints" / f"step-{step:09d}.zip"
            )
            summary = evaluate(
                checkpoint,
                cases,
                path / f"eval-{step:09d}",
                eval_ledger,
                f"{arm}@{step}",
                gait_config=GaitConfig.from_dict(protocol["evaluation_gait_config"]),
            )
            result["checkpoints"].append(
                {"steps": step, "summary": summary, "checkpoint_sha256": sha256(checkpoint)}
            )
            write_json(path / "result.json", result)
        video = replay(
            path / f"eval-{steps:09d}/seed-101-soft-0.json",
            path / "replay.mp4",
            eval_ledger,
            f"{arm} | seed {protocol['seed']} | {steps:,} steps",
        )
        result.update(
            status="complete",
            video=video,
            final_summary=result["checkpoints"][-1]["summary"],
            final_checkpoint_sha256=sha256(model),
        )
        return result
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result.update(training_budget=train_ledger.status(), evaluation_budget=eval_ledger.status())
        write_json(path / "result.json", result)


def status(folder):
    folder = Path(folder)
    result = {}
    for arm in ARMS:
        path = folder / arm
        state = (
            read_json(path / "result.json")
            if (path / "result.json").exists()
            else {"status": "pending"}
        )
        progress = (
            read_json(path / "training/progress.json")
            if (path / "training/progress.json").exists()
            else {}
        )
        result[arm] = {
            "status": state["status"],
            "error": state.get("error"),
            **progress,
            "evaluated_checkpoints": len(state.get("checkpoints", [])),
            "training_budget": Ledger(path / "training.sqlite").status(),
        }
    return result


def batch(folder):
    folder = Path(folder).resolve()
    verify_protocol(folder)

    def worker(arm):
        with (folder / arm / "worker.log").open("x") as log:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "qwop_lab.gait_study",
                    "trial",
                    "--out",
                    str(folder),
                    "--arm",
                    arm,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return arm, process.returncode

    with ThreadPoolExecutor(max_workers=2) as pool:
        exits = dict(pool.map(worker, ARMS))
    write_json(folder / "batch-exits.json", exits)
    if any(exits.values()):
        raise RuntimeError(f"Gait batch failed: {exits}; inspect preserved worker logs")
    return finish(folder)


def audit_episode(episode, config):
    if (
        digest(episode["trace"]) != episode["trace_sha256"]
        or digest(episode["actions"]) != episode["actions_sha256"]
    ):
        raise ValueError("Evaluation trace/action hash mismatch")
    if (
        len(episode["trace"]) != len(episode["actions"])
        or len(episode["trace"]) != episode["num_steps"]
    ):
        raise ValueError("Evaluation trace length mismatch")
    crossing = next((r["score_time"] for r in episode["trace"] if r["distance"] >= 100), None)
    if crossing != episode["first_100m_score_time"]:
        raise ValueError("Finish crossing mismatch")
    total = GaitAccumulator()
    previous = episode["initial"]["distance"]
    for row in [episode["initial"], *episode["trace"]]:
        if measure(row["gait_pose"], config) != row["gait"]:
            raise ValueError("Gait measurements disagree with raw pose")
    for row in episode["trace"]:
        total.add(row["gait"], row["distance"] - previous, row["reward"], 0.0)
        previous = row["distance"]
    if total.summary() != episode["gait_summary"]:
        raise ValueError("Gait summary disagrees with trajectory")
    final = episode["trace"][-1]
    if episode["is_success"] != (final["is_success"] and final["terminated"]):
        raise ValueError("Success disagrees with final transition")
    if episode["distance"] != final["distance"] or episode["score_time"] != final["score_time"]:
        raise ValueError("Final state disagrees with summary")


def finish(folder):
    folder = Path(folder).resolve()
    protocol = verify_protocol(folder)
    results = {arm: read_json(folder / arm / "result.json") for arm in ARMS}
    if not all(r["status"] == "complete" for r in results.values()):
        raise ValueError("Both arms must finish; failed arms remain visible")
    initial_hashes = []
    for arm, result in results.items():
        path = folder / arm
        meta = read_json(path / "training/manifest.json")
        if (
            meta["status"] != "complete"
            or meta["actual_steps"] != protocol["training_steps_per_arm"]
        ):
            raise ValueError("Training allocation incomplete")
        if (
            meta["gait_config"] != protocol["arms"][arm]
            or meta["ppo_config"] != protocol["ppo_config"]
        ):
            raise ValueError("Training settings differ from protocol")
        if sha256(path / "training/final.zip") != result["final_checkpoint_sha256"]:
            raise ValueError("Final model hash mismatch")
        # Zip container timestamps vary; compare serialized policy weights instead.
        from stable_baselines3 import PPO
        import torch

        weights = PPO.load(path / "training/initial.zip").policy.state_dict()
        initial_hashes.append(weights)
        budget = Ledger(path / "training.sqlite").status()
        if (
            budget["confirmed_steps"] != protocol["training_steps_per_arm"]
            or budget["charged_steps"] != budget["confirmed_steps"]
        ):
            raise ValueError("Training budget mismatch")
        expected_steps = list(
            range(
                protocol["checkpoint_interval"],
                protocol["training_steps_per_arm"] + 1,
                protocol["checkpoint_interval"],
            )
        )
        if [item["steps"] for item in result["checkpoints"]] != expected_steps:
            raise ValueError("Missing or reordered checkpoint evaluations")
        evaluation_steps = 0
        for item in result["checkpoints"]:
            evaluation = path / f"eval-{item['steps']:09d}"
            checkpoint = path / "training/checkpoints" / f"step-{item['steps']:09d}.zip"
            # The final checkpoint also has a separately serialized final.zip.
            if item["steps"] == protocol["training_steps_per_arm"]:
                checkpoint = path / "training/final.zip"
            manifest = read_json(evaluation / "manifest.json")
            if (
                sha256(checkpoint) != item["checkpoint_sha256"]
                or manifest["model"]["sha256"] != item["checkpoint_sha256"]
            ):
                raise ValueError("Evaluated model hash mismatch")
            if manifest["status"] != "complete":
                raise ValueError("Evaluation incomplete")
            episodes = [read_json(evaluation / f"seed-101-soft-{phase}.json") for phase in (0, 1)]
            for episode in episodes:
                if episode["contract"] != protocol["contract"]:
                    raise ValueError("Evaluation runtime mismatch")
                audit_episode(episode, GaitConfig.from_dict(protocol["evaluation_gait_config"]))
                evaluation_steps += episode["num_steps"]
            actual = summarize(episodes)
            actual["gait_by_case"] = {e["case"]["id"]: e["gait_summary"] for e in episodes}
            if actual != item["summary"]:
                raise ValueError("Reported checkpoint summary differs from trajectories")
        if (
            not result["video"]["verified"]
            or sha256(path / "replay.mp4") != result["video"]["video_sha256"]
        ):
            raise ValueError("Replay verification missing")
        video_trace = path / f"eval-{protocol['training_steps_per_arm']:09d}/seed-101-soft-0.json"
        if result["video"]["source_replay_sha256"] != sha256(video_trace):
            raise ValueError("Replay is not the predetermined final case")
        evaluation_steps += result["video"]["verified_transitions"]
        evaluation_budget = Ledger(path / "evaluation.sqlite").status()
        if (
            evaluation_budget["confirmed_steps"] != evaluation_steps
            or evaluation_budget["charged_steps"] != evaluation_steps
        ):
            raise ValueError("Evaluation/replay accounting mismatch")
        if result["final_summary"] != result["checkpoints"][-1]["summary"]:
            raise ValueError("Final model summary mismatch")
    for key in initial_hashes[0]:
        torch.testing.assert_close(initial_hashes[0][key], initial_hashes[1][key], rtol=0, atol=0)
    write_json(
        folder / "audit.json",
        {
            "passed": True,
            "paired_initial_weights_identical": True,
            "raw_geometry_recomputed": True,
            "final_replays_verified": True,
        },
    )
    comparison = {
        "protocol_sha256": sha256(folder / "protocol.json"),
        "arms": {arm: r["final_summary"] for arm, r in results.items()},
        "training_steps": sum(r["training_budget"]["confirmed_steps"] for r in results.values()),
        "evaluation_and_replay_steps": sum(
            r["evaluation_budget"]["confirmed_steps"] for r in results.values()
        ),
        "limitations": protocol["limitations"],
    }
    write_json(folder / "comparison.json", comparison)
    from PIL import Image, ImageDraw

    sheet = Image.new("RGB", (960, 560), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for row, arm in enumerate(ARMS):
        for column, sample in enumerate(results[arm]["video"].get("gait_samples", [])):
            source = folder / arm / sample["file"]
            if sha256(source) != sample["sha256"]:
                raise ValueError("Gait sample hash mismatch")
            im = Image.open(source).convert("RGB")
            im.thumbnail((320, 255))
            draw.text(
                (column * 320 + 5, row * 280 + 5),
                f"{arm} / {sample['fraction']:.0%} of replay",
                fill="black",
            )
            sheet.paste(im, (column * 320, row * 280 + 25))
    sheet.save(folder / "gait-samples.png")
    if not (folder / "comparison.mp4").exists():
        side_by_side(
            folder / "baseline/replay.mp4", folder / "gait/replay.mp4", folder / "comparison.mp4"
        )
    rows = [
        "# Gait reward screen",
        "",
        f"Fresh paired seed {protocol['seed']}; {protocol['training_steps_per_arm']:,} steps per arm. "
        "Both use the same 128x128 PPO configuration. Only the posture-dependent forward reward differs.",
        "",
        "| Arm / steps | Finishes | Mean distance | Upright samples | Knee near ground | Upright forward fraction | 100m game seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, result in results.items():
        for item in result["checkpoints"]:
            summary = item["summary"]
            gait = list(summary["gait_by_case"].values())
            means = [
                statistics.mean(g[k] for g in gait)
                for k in ("mean_upright", "mean_knee_near_ground", "upright_forward_fraction")
            ]
            time = summary["mean_100m_game_seconds"]
            time_label = f"{time:.3f}" if time is not None else "No finish"
            rows.append(
                f"| {arm} / {item['steps']:,} | {summary['valid_100m_finishes']}/2 | "
                f"{summary['mean_distance']:.2f} m | {means[0]:.1%} | {means[1]:.1%} | {means[2]:.1%} | {time_label} |"
            )
    rows += [
        "",
        "![Predetermined verified replay samples](gait-samples.png)",
        "",
        "Gait percentages are equal-weight means of the two episode summaries. Short failures can "
        "have high upright fractions; interpret them alongside distance and completion. Upright forward "
        "distance sums positive sampled displacement and is not net progress. These thresholds are exploratory "
        "geometric proxies, not a formal running classifier.",
        "",
        protocol["limitations"],
        "",
        f"Training used {comparison['training_steps']:,} steps; evaluation and verified replay used "
        f"{comparison['evaluation_and_replay_steps']:,}. All final models are reported.",
        "",
        "[Verified comparison video](comparison.mp4). See `audit.json`, `protocol.json`, and the per-arm "
        "raw trajectories for reproducibility. Historical diagnostics and engineering smoke tests, when run, "
        "use separate ledgers and are outside this allocation.",
        "",
    ]
    (folder / "report.md").write_text("\n".join(rows))
    return comparison


def archive(folder, destination):
    folder, destination = Path(folder).resolve(), Path(destination).resolve()
    finish(folder)
    if destination == folder or folder in destination.parents:
        raise ValueError("Archive destination must be outside the run")
    destination.mkdir(parents=True, exist_ok=False)
    for name in (
        "protocol.json",
        "protocol.sha256",
        "comparison.json",
        "audit.json",
        "report.md",
        "requirements-macos.txt",
        "gait-samples.png",
    ):
        shutil.copy2(folder / name, destination / name)
    files = [p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix != ".mp4"]
    write_json(
        destination / "artifact-index.json", {str(p.relative_to(folder)): sha256(p) for p in files}
    )
    with tarfile.open(destination / "artifacts.tar.gz", "w:gz") as bundle:
        for path in files:
            bundle.add(path, arcname=str(path.relative_to(folder)))
    video = str(folder / "comparison.mp4")
    report = (
        (destination / "report.md")
        .read_text()
        .replace(
            "[Verified comparison video](comparison.mp4)", f"[Verified comparison video]({video})"
        )
    )
    (destination / "report.md").write_text(report)
    return {"archive": str(destination)}


def main():
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "trial", "batch", "status", "finish", "archive"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", choices=tuple(ARMS))
    parser.add_argument("--steps", type=int, default=2097152)
    parser.add_argument("--interval", type=int, default=524288)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--destination")
    args = parser.parse_args()
    if args.action == "init":
        result = initialize(args.out, args.steps, args.interval, args.seed)
    elif args.action == "trial":
        if not args.arm:
            parser.error("trial requires --arm")
        result = run_trial(args.out, args.arm)
    elif args.action == "archive":
        if not args.destination:
            parser.error("archive requires --destination")
        result = archive(args.out, args.destination)
    else:
        result = {"batch": batch, "status": status, "finish": finish}[args.action](args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
