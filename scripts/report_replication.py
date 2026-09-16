"""Audit and report a completed fresh-seed architecture replication."""

import argparse
import hashlib
import io
from pathlib import Path
import shutil
import subprocess
import tarfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from qwop_lab.agents import configure_torch
from qwop_lab.artifacts import ROOT, digest, read_json, sha256, write_json
from qwop_lab.budget import Ledger
from qwop_lab.evaluation import summarize
from qwop_lab.replication import aggregate, verify_protocol
from qwop_lab.replay import side_by_side


def describe(value):
    return "No complete pair" if value is None else f"{value:.3f}"


def audit(folder, protocol, amendment=None):
    results, identities = {}, []
    for seed in protocol["seeds"]:
        for design in protocol["designs"]:
            path = folder / design / f"seed-{seed}"
            recovered = amendment and (design, seed) == (amendment["design"], amendment["seed"])
            training_dir = path / ("recovery-training" if recovered else "training")
            row = read_json(path / ("recovered-result.json" if recovered else "result.json"))
            original_manifest = read_json(path / "training/manifest.json")
            manifest = read_json(training_dir / "manifest.json")
            assert row["status"] == manifest["status"] == "complete"
            if recovered:
                assert original_manifest["status"] == "failed"
                assert (
                    manifest["actual_steps"] + original_manifest["actual_steps"]
                    == protocol["training_steps_per_run"]
                )
                assert row["amendment_sha256"] == sha256(folder / "resume-amendment.json")
            else:
                assert manifest["actual_steps"] == protocol["training_steps_per_run"]
            assert manifest["seed"] == seed
            assert manifest["ppo_config"] == protocol["ppo_config"]
            assert manifest["contract"] == protocol["contract"]
            assert manifest["architecture"] == protocol["architecture_identities"][design]
            assert original_manifest["initialization"] == "fresh_seeded_weights"
            assert original_manifest["source_checkpoint"] is None
            assert row["protocol_sha256"] == sha256(folder / "protocol.json")
            for kind in ("training", "evaluation"):
                ledger = Ledger(path / f"{kind}.sqlite").status()
                assert all(lease["closed"] for lease in ledger["leases"])
                assert ledger["charged_steps"] == ledger["confirmed_steps"]
                assert ledger["charged_steps"] <= ledger["cap"]
            state = PPO.load(path / "training/initial.zip", device="cpu").policy.state_dict()
            h = hashlib.sha256()
            for key, tensor in sorted(state.items()):
                h.update(key.encode())
                h.update(tensor.cpu().numpy().tobytes())
            identities.append(
                {"design": design, "seed": seed, "initial_state_sha256": h.hexdigest()}
            )
            for checkpoint in row["checkpoints"]:
                step = checkpoint["steps"]
                model = training_dir / "checkpoints" / f"step-{step:09d}.zip"
                if step == protocol["training_steps_per_run"]:
                    model = training_dir / "final.zip"
                assert sha256(model) == checkpoint["checkpoint_sha256"]
                meta = read_json(model.with_suffix(".json"))
                assert meta["sha256"] == sha256(model)
                assert meta["trained_steps"] == step
                evaluation = path / f"eval-{step:09d}"
                eval_meta = read_json(evaluation / "manifest.json")
                assert eval_meta["status"] == "complete"
                assert eval_meta["model"]["sha256"] == sha256(model)
                episodes = []
                for case in protocol["evaluation_cases"]:
                    episode = read_json(
                        evaluation / f"seed-{case['seed']}-soft-{case['soft_resets']}.json"
                    )
                    assert episode["contract"] == protocol["contract"]
                    assert digest(episode["trace"]) == episode["trace_sha256"]
                    assert digest(episode["actions"]) == episode["actions_sha256"]
                    assert len(episode["actions"]) == len(episode["trace"]) == episode["num_steps"]
                    crossing = next(
                        (s["score_time"] for s in episode["trace"] if s["distance"] >= 100), None
                    )
                    assert crossing == episode["first_100m_score_time"]
                    final = episode["trace"][-1]
                    assert episode["is_success"] == (final["is_success"] and final["terminated"])
                    episodes.append(episode)
                assert (
                    summarize(episodes)
                    == checkpoint["summary"]
                    == read_json(evaluation / "summary.json")
                )
            assert row["final_summary"] == row["checkpoints"][-1]["summary"]
            video = read_json(path / "replay.json")
            assert video["verified"]
            assert video["video_sha256"] == sha256(path / "replay.mp4")
            assert video["source_replay_sha256"] == sha256(
                path / f"eval-{protocol['training_steps_per_run']:09d}/seed-101-soft-0.json"
            )
            results[(design, seed)] = row
    for design in protocol["designs"]:
        assert len({r["initial_state_sha256"] for r in identities if r["design"] == design}) == len(
            protocol["seeds"]
        )
    return results, identities


def plots(results, protocol, output, amendment=None):
    colors = {"baseline": "#266eaa", "kinematic": "#15836b"}
    fig, axes = plt.subplots(
        2, 3, figsize=(13, 7.8), sharex=True, sharey="row", constrained_layout=True
    )
    for column, seed in enumerate(protocol["seeds"]):
        for design in protocol["designs"]:
            checkpoints = results[(design, seed)]["checkpoints"]
            x = [c["steps"] / 1e6 for c in checkpoints]
            finishes = [c["summary"]["valid_100m_finishes"] for c in checkpoints]
            times = [
                c["summary"]["mean_100m_game_seconds"]
                if c["summary"]["valid_100m_finishes"] == 2
                else np.nan
                for c in checkpoints
            ]
            axes[0, column].plot(x, finishes, marker="o", color=colors[design], label=design)
            axes[1, column].plot(x, times, marker="o", color=colors[design], label=design)
        title = f"Training seed {seed}"
        if amendment and seed == amendment["seed"]:
            title += " (baseline resumed)"
            for ax in axes[:, column]:
                ax.axvline(
                    amendment["spent_training_steps"] / 1e6, ls=":", color="#888888", alpha=0.6
                )
        axes[0, column].set(title=title, yticks=[0, 1, 2], ylim=(-0.1, 2.1))
        axes[1, column].set(xlabel="Training interactions (millions)")
    axes[0, 0].set_ylabel("Reset phases finished (out of 2)")
    axes[1, 0].set_ylabel("Mean 100m game seconds\n(only when both phases finish)")
    axes[0, 0].legend()
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("QWOP architecture replication — every seed and fixed checkpoint", fontsize=15)
    fig.savefig(output / "learning-curves.png", dpi=170)
    fig.savefig(output / "learning-curves.svg")
    plt.close(fig)


def contact_sheet(folder, protocol, output):
    cell_w, cell_h = 320, 286
    sheet = Image.new("RGB", (3 * cell_w, 6 * cell_h), "#eef1f5")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
    samples = []
    row = 0
    for seed in protocol["seeds"]:
        for design in protocol["designs"]:
            path = folder / design / f"seed-{seed}"
            meta = read_json(path / "replay.json")
            duration = meta["duration_seconds"] - meta["final_hold_seconds"]
            for column, fraction in enumerate((0.2, 0.5, 0.8)):
                at = max(0, fraction * duration)
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-ss",
                        str(at),
                        "-i",
                        str(path / "replay.mp4"),
                        "-frames:v",
                        "1",
                        "-f",
                        "image2pipe",
                        "-vcodec",
                        "png",
                        "-",
                    ],
                    capture_output=True,
                    check=True,
                )
                frame = Image.open(io.BytesIO(result.stdout)).convert("RGB")
                frame.thumbnail((cell_w, cell_h - 35))
                x, y = column * cell_w, row * cell_h
                sheet.paste(frame, (x + (cell_w - frame.width) // 2, y + 35))
                draw.text(
                    (x + 8, y + 8),
                    f"{design} / {seed} / {at:.1f}s video",
                    font=font,
                    fill="#203043",
                )
                samples.append(
                    {"design": design, "seed": seed, "video_seconds": at, "fraction": fraction}
                )
            row += 1
    sheet.save(output / "gait-contact-sheet.png")
    write_json(
        output / "gait-samples.json",
        {"method": "20%, 50%, 80% of each fixed-case final replay", "samples": samples},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gait-review", required=True)
    args = parser.parse_args()
    folder, output = Path(args.campaign).resolve(), Path(args.out).resolve()
    comparison = read_json(folder / "comparison.json")
    amendment = comparison.get("amendment")
    if amendment:
        from qwop_lab.replication_resume import verify

        protocol, checked_amendment = verify(folder)
        assert amendment == checked_amendment
    else:
        protocol = verify_protocol(folder)
    configure_torch()
    results, identities = audit(folder, protocol, amendment)
    assert comparison["summary"] == aggregate(results, protocol["seeds"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "audit.json", {"passed": True, "initial_policy_identities": identities})
    write_json(output / "comparison.json", comparison)
    shutil.copy2(folder / "protocol.json", output / "protocol.json")
    shutil.copy2(folder / "protocol.sha256", output / "protocol.sha256")
    plots(results, protocol, output, amendment)
    contact_sheet(folder, protocol, output)
    gait = read_json(args.gait_review)
    write_json(output / "gait-review.json", gait)
    diagnostic_image = folder / "baseline/seed-61/checkpoint-786432-gait-late/samples.png"
    if diagnostic_image.exists():
        shutil.copy2(diagnostic_image, output / "intermediate-stall.png")
    for seed in protocol["seeds"]:
        video = folder / f"comparison-seed-{seed}.mp4"
        if not video.exists():
            side_by_side(
                folder / f"baseline/seed-{seed}/replay.mp4",
                folder / f"kinematic/seed-{seed}/replay.mp4",
                video,
            )
    costs = {**comparison["costs"]}
    costs["posthoc_diagnostic_replay_steps"] = sum(
        read_json(path)["verified_prefix_steps"] for path in folder.rglob("diagnostic.json")
    )
    startup_paths = sorted((ROOT / "runs").glob("replication-startup-*.sqlite"))
    costs["separate_startup_checks"] = {
        p.name: Ledger(p).status()["charged_steps"] for p in startup_paths
    }
    costs["failed_launch_training_steps"] = protocol.get("startup_recovery", {}).get(
        "previous_charged_steps", 0
    )
    failed_start_folder = Path(protocol["startup_recovery"]["previous_folder"])
    costs["failed_startup_process_seconds"] = sum(
        read_json(path)["wall_seconds"]
        for path in failed_start_folder.glob("*/seed-*/training/manifest.json")
    )
    write_json(output / "costs.json", costs)
    summary = comparison["summary"]
    lines = [
        "# QWOP architecture replication",
        "",
        "Three fresh training seeds per design, with 1,048,576 interactions for every run. "
        "Baseline: 128x128 Tanh PPO. Kinematic: the previously selected 64x64 PPO design with "
        "fixed torso-relative position, velocity, and periodic angle features. "
        "All learning settings, reward, action mapping, game physics, and evaluation rules are shared.",
        "",
        "## Every final model",
        "",
        "| Training seed | Baseline finishes | Baseline mean seconds | Kinematic finishes | Kinematic mean seconds | Kinematic time change |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in summary["pairs"]:
        a, b = pair["baseline"], pair["kinematic"]
        change = pair["kinematic_time_change_percent"]
        delta = "Not comparable" if change is None else f"{change:+.2f}%"
        seed_label = str(pair["seed"]) + (
            "*" if amendment and pair["seed"] == amendment["seed"] else ""
        )
        lines.append(
            f"| {seed_label} | {a['finishes']}/2 | {describe(a['both_phase_mean_100m_game_seconds'])} | {b['finishes']}/2 | {describe(b['both_phase_mean_100m_game_seconds'])} | {delta} |"
        )
    lines += [
        "",
        "Times use the game's clock at the first observed 100m crossing, conditional on "
        "native success. Mean times require both reset phases to finish. Negative percentage "
        "changes favor the kinematic design. Every training seed is retained; failures are visible.",
        "",
    ]
    for design, row in summary["designs"].items():
        lines.append(
            f"- **{design}:** {row['runs_finishing_both_phases']}/{row['training_runs']} training runs finished both phases; {row['phase_finishes']}/{row['phase_episodes']} phase finishes. Mean time among runs finishing both phases: {describe(row['mean_time_among_both_phase_finishing_runs'])} seconds. Parameters: {results[(design, protocol['seeds'][0])]['trainable_parameters']:,}."
        )
    if summary["mean_paired_time_change_percent"] is not None:
        lines += [
            "",
            f"Mean paired time change: **{summary['mean_paired_time_change_percent']:+.2f}%** across {summary['comparable_speed_pairs']} fully finishing seed pairs. Three seed pairs are a small descriptive sample, not strong statistical evidence.",
        ]
    if amendment:
        sensitivity = comparison["uninterrupted_seed_sensitivity"]
        lines += [
            "",
            "**Recorded interruption:** baseline training seed 29 resumed from its saved weights "
            "and optimizer after a Windows progress-file replacement failed. Its game and action RNG "
            f"were reset; {amendment['discarded_rollout_interactions']:,} unoptimized rollout interactions "
            "remain charged. All models spent the same total interactions, but this one run is not "
            "an uninterrupted replication. Its original failure is preserved.",
            "",
            f"Sensitivity analysis using only uninterrupted seed pairs (17 and 61): "
            f"{sensitivity['comparable_speed_pairs']} fully finishing pairs; mean paired time change "
            f"{describe(sensitivity['mean_paired_time_change_percent'])}%. "
            "Reliability counts and all individual seed results remain in comparison.json.",
        ]
    lines += [
        "",
        "## Learning curves",
        "",
        "![Learning curves](learning-curves.png)",
        "",
        "Intermediate checkpoints are diagnostic; all final models are evaluated at the same fixed budget. "
        "A missing time point means at least one reset phase failed, not zero seconds.",
        "",
        "## Movement",
        "",
        gait["summary"],
        "",
        "![Fixed replay samples](gait-contact-sheet.png)",
        "",
        "Qualitative visual assessment of predetermined replay samples. No formal gait classifier is used.",
        "",
    ]
    if diagnostic_image.exists():
        lines += [
            "",
            "### Intermediate startup stall",
            "",
            "Baseline seed 61's 786,432-step checkpoint took 16.24 game seconds to reach its first metre "
            "in phase zero, then crossed 100m at 30.26 seconds. Verified diagnostic frames show an "
            "initial near-stationary lunge followed by knee-scooting. This explains the large slowdown "
            "in its learning curve. The checkpoint remains diagnostic and is not substituted for its final model.",
            "",
            "![Intermediate stall diagnosis](intermediate-stall.png)",
            "",
            f"These post-hoc diagnostic prefixes used {costs['posthoc_diagnostic_replay_steps']:,} additional "
            "replay interactions within the existing evaluation cap; they are included in the total cost.",
            "",
        ]
    lines += ["## Verified videos", ""]
    for seed in protocol["seeds"]:
        lines.append(
            f"- [Baseline versus kinematic, training seed {seed}]({(folder / f'comparison-seed-{seed}.mp4').as_posix()})"
        )
    lines += [
        "",
        "## Limits and interpretation",
        "",
        "This tests a previously selected architecture across new training randomness. It does not repeat "
        "the AI proposal process, compare AI versus ordinary search, or establish architectural novelty. "
        "The kinematic design combines a different feature representation with smaller policy/value heads; "
        "these effects are not isolated. All policies use PPO; QR-DQN has not been tested here.",
        "",
        "There are three independent training runs per design. The two evaluation reset phases have very "
        "limited physical diversity and are not a held-out robustness suite. Extra episode labels would not "
        "create extra independent training evidence. Faster knee-scooting would not establish convincing running.",
        "",
        "## Costs and startup recovery",
        "",
        f"Training: **{costs['training_steps']:,} interactions**. Evaluation and replay: **{costs['evaluation_and_replay_steps']:,}**. "
        f"Total experiment: **{costs['total_steps']:,}**. Summed training-process duration: {costs['summed_training_process_seconds'] / 3600:.2f} hours; "
        "processes overlapped, so this is not elapsed wall time or CPU-seconds. Separate startup diagnostics appear in costs.json.",
        f"The preceding zero-step startup failures additionally consumed {costs['failed_startup_process_seconds'] / 60:.2f} summed process minutes.",
        "",
        "The preceding launch batch failed all six starts before taking any game steps. Its artifacts and zero-step "
        "ledgers are preserved and hash-linked by this protocol. Clearing orphaned headless browsers restored startup. "
        "Browser cleanup and registration timeout handling were adjusted before freezing this replacement batch. "
        "The six planned training allowances were not expanded. The Python runner made no model API calls; Codex usage is not instrumented here.",
        "",
        "## Reproducibility",
        "",
        "The audit recomputes scores from all saved trajectories, checks model and trace hashes, verifies configuration "
        "and step budgets, confirms distinct initial weights across training seeds, and verifies the six final videos. "
        "The archive contains frozen source, models, evaluations, ledgers, and preserved failed-start artifacts. "
        "The proprietary game and browser are excluded; compatible bootstrapped runtime files are required.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    files = {}
    excluded = {".mp4", ".png"}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix not in excluded and path.name != "running.lock":
            files[f"run/{path.relative_to(folder).as_posix()}"] = path
    previous = Path(protocol["startup_recovery"]["previous_folder"])
    for path in sorted(previous.rglob("*")):
        if path.is_file() and path.name != "running.lock":
            files[f"failed-start/{path.relative_to(previous).as_posix()}"] = path
    for path in output.iterdir():
        if path.is_file():
            files[f"report/{path.name}"] = path
    for path in [
        Path(__file__),
        ROOT / "scripts/check_runtime.py",
        ROOT / "scripts/verify_saved_trace.py",
        ROOT / "scripts/evaluate_replication_checkpoint.py",
        ROOT / "scripts/read_replication_status.py",
        ROOT / "scripts/resume_when_available.py",
        ROOT / "scripts/inspect_trace_prefix.py",
        ROOT / "scripts/cleanup_orphaned_qwop_browsers.ps1",
        ROOT / "scripts/bootstrap.ps1",
        ROOT / "pyproject.toml",
        ROOT / "requirements.lock.txt",
        ROOT / "checkpoints.lock.json",
        ROOT / "game-source.lock.json",
        ROOT / ".runtime/runtime.json",
        *sorted((ROOT / "tests").glob("*.py")),
    ]:
        files[f"support/{path.relative_to(ROOT).as_posix()}"] = path
    for path in startup_paths:
        files[f"startup-checks/{path.name}"] = path
    if amendment:
        files["source-amendment/qwop_lab/replication_resume.py"] = (
            ROOT / "qwop_lab/replication_resume.py"
        )
        for relative in amendment["source_changes"]:
            files[f"source-amendment/{relative}"] = ROOT / relative
    index = {name: sha256(path) for name, path in files.items()}
    archive = output / "artifacts.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, path in sorted(files.items()):
            bundle.add(path, arcname=name)
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            stream = bundle.extractfile(member)
            assert hashlib.file_digest(stream, "sha256").hexdigest() == index[member.name]
    write_json(
        output / "artifact-index.json",
        {
            "archive_sha256": sha256(archive),
            "files": index,
            "videos": {
                str(path.relative_to(folder)): sha256(path) for path in folder.rglob("*.mp4")
            },
        },
    )
    print(f"Verified {len(index)} archived files; report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
