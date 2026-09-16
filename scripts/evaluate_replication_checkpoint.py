"""Evaluate an already saved early checkpoint while its training run continues.

Uses the frozen cases and existing evaluation allowance. The main worker reuses
the verified evaluation after training, so this adds no evaluations or budget.
"""

import argparse
import multiprocessing
from pathlib import Path
from read_replication_status import read_shared_json

from qwop_lab.artifacts import read_json, sha256, write_json
from qwop_lab.budget import Ledger
from qwop_lab.environment import Case
from qwop_lab.evaluation import evaluate
from qwop_lab.replication import verify_protocol


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--step", required=True, type=int)
    args = parser.parse_args()
    folder = Path(args.campaign).resolve()
    protocol = verify_protocol(folder)
    assert args.design in protocol["designs"] and args.seed in protocol["seeds"]
    assert args.step in range(
        protocol["checkpoint_interval"],
        protocol["training_steps_per_run"],
        protocol["checkpoint_interval"],
    )
    path = folder / args.design / f"seed-{args.seed}"
    progress = read_shared_json(path / "training/progress.json")
    manifest = read_json(path / "training/manifest.json")
    assert manifest["status"] == "running"
    assert args.step < progress["steps"] < protocol["training_steps_per_run"] - 131072
    model = path / "training/checkpoints" / f"step-{args.step:09d}.zip"
    meta = read_json(model.with_suffix(".json"))
    assert meta["trained_steps"] == args.step and meta["sha256"] == sha256(model)
    output = path / f"eval-{args.step:09d}"
    summary = evaluate(
        model,
        [Case(**case) for case in protocol["evaluation_cases"]],
        output,
        Ledger(path / "evaluation.sqlite"),
        f"{args.design}/seed-{args.seed}@{args.step}",
    )
    write_json(
        output / "early-evaluation.json",
        {
            "checkpoint_sha256": sha256(model),
            "protocol_sha256": sha256(folder / "protocol.json"),
            "training_steps_at_dispatch": progress["steps"],
            "additional_budget": 0,
            "note": "Evaluation of a planned checkpoint performed early; main worker reuses it",
        },
    )
    print(summary)
