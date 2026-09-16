"""Build an offline report and a hash-indexed archive after a completed campaign."""

import argparse
import json
import shutil
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qwop_lab.artifacts import ROOT, read_json, sha256, write_json
from qwop_lab.budget import Ledger
from qwop_lab.replay import side_by_side


def describe(value, places=3):
    return "No finish" if value is None else f"{value:.{places}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    campaign, output = Path(args.campaign).resolve(), Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = read_json(campaign / "comparison.json")
    protocol = report["protocol"]
    effective_arms = report.get("effective_arms", protocol["arms"])
    trials = []
    for arm, slots in effective_arms.items():
        for slot in slots:
            trial = read_json(campaign / arm / slot / "result.json")
            meta = read_json(campaign / arm / slot / "training" / "manifest.json")
            trials.append((arm, slot, trial, meta))
    trained_total = sum(meta["actual_steps"] for _, _, _, meta in trials)
    training_total = sum(
        Ledger(campaign / arm / "training.sqlite").status()["confirmed_steps"]
        for arm in protocol["arms"]
    )
    failed_meta = None
    if report.get("amendment"):
        failed_meta = read_json(campaign / "research/proposal-1/training/manifest.json")
    evaluation_total = sum(
        Ledger(campaign / arm / "evaluation.sqlite").status()["confirmed_steps"]
        for arm in protocol["arms"]
    )
    costs = {
        "training_steps": training_total,
        "training_steps_in_completed_trials": trained_total,
        "training_steps_in_failed_attempts": training_total - trained_total,
        "evaluation_and_replay_steps": evaluation_total,
        "total_campaign_steps": training_total + evaluation_total,
        "summed_training_wall_seconds": sum(meta["wall_seconds"] for _, _, _, meta in trials)
        + (failed_meta["wall_seconds"] if failed_meta else 0),
        "wall_time_note": "Training processes overlap; summed durations are not elapsed campaign time or CPU-seconds.",
    }
    smoke = ROOT / "runs" / "architecture-smoke" / "budget.sqlite"
    if smoke.exists():
        costs["separate_engineering_smoke_steps"] = Ledger(smoke).status()["confirmed_steps"]
    write_json(output / "costs.json", costs)
    write_json(output / "comparison.json", report)
    shutil.copy2(campaign / "protocol.json", output / "protocol.json")
    if report.get("amendment"):
        shutil.copy2(campaign / "amendment-001.json", output / "amendment-001.json")

    colors = {"baseline": "#2374ab", "search": "#9d6a24", "research": "#21866d"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    has_finish = False
    for index, (arm, slot, trial, _) in enumerate(trials):
        x = [entry["steps"] / 1e6 for entry in trial["checkpoints"]]
        distance = [entry["summary"]["mean_distance"] for entry in trial["checkpoints"]]
        label = f"{arm}: {slot}"
        style = "--" if slot in ("large", "proposal-2") else "-"
        axes[0].plot(x, distance, style, marker="o", color=colors[arm], label=label)
        finished = [
            (entry["steps"] / 1e6, entry["summary"]["best_100m_game_seconds"])
            for entry in trial["checkpoints"]
            if entry["summary"]["best_100m_game_seconds"] is not None
        ]
        if finished:
            has_finish = True
            axes[1].plot(*zip(*finished), style, marker="o", color=colors[arm], label=label)
    axes[0].set(ylabel="Mean terminal distance (m)", title="Development progress")
    axes[0].axhline(100, color="#999999", lw=1, alpha=0.6)
    axes[0].legend(fontsize=8)
    axes[1].set(
        ylabel="Best valid 100m game time (seconds)", title="Finish-line speed (lower is better)"
    )
    if not has_finish:
        axes[1].text(0.5, 0.5, "No valid finishes", transform=axes[1].transAxes, ha="center")
    for ax in axes:
        ax.set_xlabel("Training steps within each trial (millions)")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("QWOP architecture screen — one training seed, two development reset phases")
    fig.savefig(output / "learning-curves.png", dpi=170)
    fig.savefig(output / "learning-curves.svg")
    plt.close(fig)

    comparison_video = campaign / "comparison.mp4"
    if not comparison_video.exists():
        side_by_side(
            campaign / "baseline" / "replay.mp4",
            campaign / "research" / "replay.mp4",
            comparison_video,
        )

    lines = [
        "# QWOP architecture campaign 001",
        "",
        "A completed exploratory campaign comparing a fixed PPO baseline, a two-candidate "
        "predefined MLP search, and two researcher-authored architecture proposals. "
        "All approaches had an allowance of 1,048,576 fresh-training interactions with the same PPO settings. "
        "One researcher in the current Codex task chose the AI proposals; the Python runner "
        "made no model API calls.",
        "",
        "## Selected final policies",
        "",
        "| Approach | Selected trial | Best 100m game seconds | Mean 100m game seconds | Valid finishes | Parameters |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for arm, comparison in report["comparisons"].items():
        s = comparison["summary"]
        lines.append(
            f"| {arm} | {comparison['selected_slot']} | {describe(s['best_100m_game_seconds'])} "
            f"| {describe(s['mean_100m_game_seconds'])} | {s['valid_100m_finishes']}/{s['episodes']} "
            f"| {comparison['trainable_parameters']:,} |"
        )
    reference = campaign / "reference/speed.json"
    if reference.exists():
        historical = read_json(reference)["summary"]
        lines += [
            "",
            f"Historical speed-policy context, rescored from saved trajectories: "
            f"{historical['best_100m_game_seconds']:.3f} best / "
            f"{historical['mean_100m_game_seconds']:.3f} mean game seconds to 100m, "
            f"{historical['valid_100m_finishes']}/{historical['episodes']} valid finishes. "
            "Its original training budget is not matched to this campaign; this is a performance reference.",
        ]
    baseline = report["comparisons"]["baseline"]["summary"]["best_100m_game_seconds"]
    research = report["comparisons"]["research"]["summary"]["best_100m_game_seconds"]
    search = report["comparisons"]["search"]["summary"]["best_100m_game_seconds"]
    if all(value is not None for value in (baseline, research, search)):
        interpretation = {
            "research_time_change_percent_vs_fixed": 100 * (research / baseline - 1),
            "research_time_reduction_percent_vs_predefined_search": 100 * (1 - research / search),
            "research_parameter_reduction_percent_vs_fixed": 100
            * (
                1
                - report["comparisons"]["research"]["trainable_parameters"]
                / report["comparisons"]["baseline"]["trainable_parameters"]
            ),
            "claim_limit": "One amended exploratory campaign; differences do not establish general researcher superiority.",
        }
        write_json(output / "interpretation.json", interpretation)
        fixed_gap = interpretation["research_time_change_percent_vs_fixed"]
        lines += [
            "",
            "## Interpretation",
            "",
            f"The selected research design is {abs(fixed_gap):.2f}% "
            f"{'slower' if fixed_gap > 0 else 'faster'} than the fixed baseline by best observed "
            "100m time. Its difference versus predefined search and parameter count are saved "
            "in `interpretation.json`. These are descriptive results from one training seed, "
            "not statistical evidence that AI research beats conventional methods.",
        ]
    lines += [
        "",
        "Game seconds are raw info.time at the first observed 100m crossing, conditional on "
        "native terminal success. Old pilot reports used a different clock scale and native "
        "termination time. Video playback follows simulation time, about ten times game seconds.",
        "",
        "## Every trial",
        "",
        "| Approach / trial | Steps | Parameters | Training minutes | Best development 100m seconds | Development finishes | Mean terminal distance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, slot, trial, meta in trials:
        s = trial["final_summary"]
        lines.append(
            f"| {arm}/{slot} | {meta['actual_steps']:,} | {meta['trainable_parameters']:,} "
            f"| {meta['wall_seconds'] / 60:.1f} | {describe(s['best_100m_game_seconds'])} "
            f"| {s['valid_100m_finishes']}/{s['episodes']} | {s['mean_distance']:.2f} m |"
        )
    if failed_meta:
        lines += [
            f"| research/proposal-1 (transport failure) | {failed_meta['actual_steps']:,} "
            f"| {failed_meta['trainable_parameters']:,} | {failed_meta['wall_seconds'] / 60:.1f} "
            "| Not evaluated | Not evaluated | Not evaluated |",
            "",
            "**Protocol amendment:** The first research trial lost its browser response. The "
            "failed attempt remains in the archive and budget. An identical architecture restarted "
            f"from fresh weights for {report['amendment']['retry_steps']:,} steps using only the "
            "original slot's remaining allowance. The retry received less effective training; "
            f"{report['amendment']['unused_rounding_steps']} steps remained unused to preserve complete "
            "PPO rollouts. The original protocol and the explicit recovery amendment are both archived.",
        ]
    lines += ["", "![Learning curves](learning-curves.png)", "", "## Research decisions", ""]
    for arm, slot, _, _ in trials:
        if arm != "research":
            continue
        proposal = read_json(campaign / arm / slot / "proposal.json")
        lines += [f"### {slot}", "", proposal["hypothesis"], "", proposal["expected_signal"], ""]
        if proposal.get("decision"):
            lines += [proposal["decision"], ""]
    lines += [
        "## Costs and limitations",
        "",
        f"Campaign environment steps: **{costs['total_campaign_steps']:,}**, including "
        f"{training_total:,} training and {evaluation_total:,} evaluation/replay steps. "
        f"Summed training-process duration: {costs['summed_training_wall_seconds'] / 3600:.2f} hours "
        "(processes overlapped).",
        "",
    ]
    if "separate_engineering_smoke_steps" in costs:
        lines += [
            f"Separate, disclosed engineering smoke test: {costs['separate_engineering_smoke_steps']:,} steps.",
            "",
        ]
    lines += [f"- {item}" for item in protocol["comparison_limits"]]
    lines += [
        "",
        "The final six cases mostly repeat two physical reset phases. They are not "
        "independent training replications or a held-out robustness test. Any apparent "
        "advantage needs fresh-seed replication before a superiority claim.",
        "",
        "## Verified replays",
        "",
        f"- [Baseline versus AI research]({comparison_video.as_posix()})",
        f"- [Ordinary search]({(campaign / 'search/replay.mp4').as_posix()})",
        "",
        "## Reproducibility",
        "",
        "`artifacts.tar.gz` includes proposals, frozen source, policy checkpoints and their "
        "architecture source, manifests, budgets and raw evaluation traces. `artifact-index.json` "
        "lists their SHA-256 hashes. Proprietary game code, browser files and videos are excluded. "
        "Use the repository's locked bootstrap for the game runtime; compatible browser/runtime "
        "hashes remain necessary for deterministic replay.",
        "",
    ]
    gait_path = campaign / "gait-review.json"
    if gait_path.exists():
        gait = read_json(gait_path)
        lines += ["## Gait inspection", "", gait["observation"], "", gait["method_limit"], ""]
        shutil.copy2(gait_path, output / "gait-review.json")
        contact = campaign / "gait-contact-sheet.png"
        if contact.exists():
            shutil.copy2(contact, output / "gait-contact-sheet.png")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")

    entries = {}
    archive = output / "artifacts.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(campaign.rglob("*")):
            if (
                not path.is_file()
                or path.suffix in (".mp4", ".png", ".pyc")
                or "__pycache__" in path.parts
            ):
                continue
            name = "campaign/" + path.relative_to(campaign).as_posix()
            bundle.add(path, arcname=name)
            entries[name] = sha256(path)
        for name in (
            "pyproject.toml",
            "requirements.lock.txt",
            "game-source.lock.json",
            "README.md",
            "RESEARCH_PLAN.md",
        ):
            path = ROOT / name
            bundle.add(path, arcname="setup/" + name)
            entries["setup/" + name] = sha256(path)
        # Include newly authored second-trial modules alongside the frozen starting source.
        for path in sorted((ROOT / "qwop_lab" / "candidates").glob("*.py")):
            name = "candidate-source/" + path.name
            bundle.add(path, arcname=name)
            entries[name] = sha256(path)
        for path in sorted((ROOT / "tests").glob("*.py")):
            name = "verification/tests/" + path.name
            bundle.add(path, arcname=name)
            entries[name] = sha256(path)
        bundle.add(__file__, arcname="verification/report_campaign.py")
        entries["verification/report_campaign.py"] = sha256(__file__)
    write_json(
        output / "artifact-index.json",
        {
            "archive_sha256": sha256(archive),
            "files": entries,
            "videos": {arm: sha256(campaign / arm / "replay.mp4") for arm in protocol["arms"]},
        },
    )
    print(
        json.dumps(
            {"report": str(output / "report.md"), "costs": costs, "archive_files": len(entries)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
