"""A fixed, small two-arm experiment. No automatic search or winner selection."""

from pathlib import Path

from .artifacts import read_json, write_json
from .budget import Ledger
from .environment import VALIDATION_CASES
from .evaluation import evaluate
from .replay import replay, side_by_side
from .training import train


def finish_pilot(folder):
    folder = Path(folder)
    arms = {name: read_json(folder / name / "manifest.json") for name in ("control", "treatment")}
    if any(arm["status"] != "complete" for arm in arms.values()):
        raise ValueError("Both training arms must be complete")
    for key in (
        "source_checkpoint_sha256",
        "seed",
        "actual_steps",
        "ppo_config",
        "architecture",
        "contract",
    ):
        if arms["control"][key] != arms["treatment"][key]:
            raise ValueError(f"Training arms differ unexpectedly: {key}")
    for module in ("training", "agents", "environment", "browser", "budget", "artifacts"):
        key = f"qwop_lab/{module}.py"
        if (
            arms["control"]["provenance"]["source_hashes"][key]
            != arms["treatment"]["provenance"]["source_hashes"][key]
        ):
            raise ValueError(f"Training implementation changed between arms: {module}")
    summaries = {name: read_json(folder / f"{name}-eval" / "summary.json") for name in arms}
    for name, arm in arms.items():
        evaluation = read_json(folder / f"{name}-eval" / "manifest.json")
        if evaluation["status"] != "complete":
            raise ValueError(f"Evaluation is incomplete: {name}")
        if evaluation["model"]["sha256"] != arm["final_checkpoint_sha256"]:
            raise ValueError(f"Evaluation used a different checkpoint: {name}")
    pairs = []
    for case in VALIDATION_CASES:
        episodes = {name: read_json(folder / f"{name}-eval" / f"{case.id}.json") for name in arms}
        if episodes["control"]["contract"] != episodes["treatment"]["contract"]:
            raise ValueError("Evaluation contracts differ")
        pairs.append(
            {
                "case": case.id,
                **{
                    name: {k: episode[k] for k in ("is_success", "score_time", "distance")}
                    for name, episode in episodes.items()
                },
            }
        )
    both_finish_all = all(s["finishes"] == s["episodes"] for s in summaries.values())
    improvement = None
    if both_finish_all:
        improvement = 100 * (
            1
            - summaries["treatment"]["mean_finish_score_time"]
            / summaries["control"]["mean_finish_score_time"]
        )
    report = {
        "design": "Single-training-seed pilot; time penalty 10 vs 30; all other configured settings matched",
        "summaries": summaries,
        "paired_cases": pairs,
        "finish_time_reduction_percent_if_both_finish_all": improvement,
        "budget": Ledger(folder / "budget.sqlite").status(),
        "limitations": [
            "One training seed, no uncertainty estimate for training variability",
            "Seed labels do not imply distinct initial physical states",
            "Development validation only; no held-out final test",
            "No claim of AI research superiority; no search-control campaign",
        ],
    }
    write_json(folder / "comparison.json", report)
    return report


def pilot(checkpoint, folder, steps=131072, seed=42):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    # 2 training arms, 12 worst-case evaluations, 2 worst-case replays.
    ledger = Ledger(folder / "budget.sqlite", 2 * steps + 70000)
    write_json(
        folder / "plan.json",
        {
            "steps_per_arm": steps,
            "training_seed": seed,
            "control_time_cost": 10,
            "treatment_time_cost": 30,
            "budget_cap": 2 * steps + 70000,
            "video_case": VALIDATION_CASES[0].id,
        },
    )
    for name, penalty in (("control", 10), ("treatment", 30)):
        model = train(checkpoint, folder / name, ledger, steps, penalty, seed)
        evaluate(model, VALIDATION_CASES, folder / f"{name}-eval", ledger, name)
    for name in ("control", "treatment"):
        replay(
            folder / f"{name}-eval" / f"{VALIDATION_CASES[0].id}.json",
            folder / f"{name}.mp4",
            ledger,
            f"PPO {name}: time penalty {10 if name == 'control' else 30}",
        )
    side_by_side(folder / "control.mp4", folder / "treatment.mp4", folder / "comparison.mp4")
    return finish_pilot(folder)
