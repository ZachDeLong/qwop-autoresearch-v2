"""Inspect measured gait against the existing speed policy before study freeze."""

import argparse
import multiprocessing
from pathlib import Path

from PIL import Image

from qwop_lab.artifacts import write_json
from qwop_lab.budget import Ledger
from qwop_lab.environment import Case, make_env, reset_case
from qwop_lab.evaluation import evaluate
from qwop_lab.gait import GaitConfig, GaitWrapper
from qwop_lab.replay import replay


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=False)
    ledger = Ledger(folder / "budget.sqlite", 15000)
    env = None
    try:
        env = GaitWrapper(make_env(), GaitConfig())
        obs, info = reset_case(env, Case(101))
        write_json(
            folder / "initial-geometry.json",
            {
                "distance": float(info["distance"]),
                "gait": info["gait"],
                "gait_pose": info["gait_pose"],
            },
        )
        Image.fromarray(env.render()).save(folder / "initial.png")
    finally:
        if env:
            env.close()
    evaluate(
        ".runtime/checkpoints/speed.pt",
        [Case(101, phase) for phase in (0, 1)],
        folder / "evaluation",
        ledger,
        "historical gait diagnostic",
        gait_config=GaitConfig(),
    )
    replay(
        folder / "evaluation/seed-101-soft-0.json",
        folder / "replay.mp4",
        ledger,
        "Historical speed policy | gait diagnostic",
    )
