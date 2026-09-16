"""Compact, read-only progress from the two gait-study workers.

Episode statistics describe stochastic training, not deterministic evaluation.
Only complete JSONL records are used while workers are still appending.
"""

import argparse
from collections import deque
import json
from pathlib import Path
import statistics


def summarize(folder, window=20):
    folder = Path(folder)
    protocol = json.loads((folder / "protocol.json").read_text())
    result = {
        "episode_statistics": "Recent stochastic training episodes; not validation",
        "arms": {},
    }
    for arm in protocol["arms"]:
        path = folder / arm
        progress = (
            json.loads((path / "training/progress.json").read_text())
            if (path / "training/progress.json").exists()
            else {}
        )
        state = (
            json.loads((path / "result.json").read_text())
            if (path / "result.json").exists()
            else {"status": "pending"}
        )
        log = path / "training/episodes.jsonl"
        episodes = []
        if log.exists():
            with log.open() as stream:
                episodes = [
                    json.loads(line)
                    for line in deque(stream, maxlen=window + 1)
                    if line.endswith("\n")
                ][-window:]
        steps = progress.get("steps", 0)
        row = {
            "status": state["status"],
            "error": state.get("error"),
            "steps": steps,
            "target": protocol["training_steps_per_arm"],
            "percent": round(100 * steps / protocol["training_steps_per_arm"], 1),
            "steps_per_second": round(progress.get("steps_per_second", 0)),
            "evaluated_checkpoints": len(state.get("checkpoints", [])),
        }
        if episodes:
            row["recent_training"] = {
                "episodes": len(episodes),
                "last_episode_step": episodes[-1]["step"],
                "mean_distance_m": round(statistics.mean(e["distance"] for e in episodes), 2),
                "finishes": sum(e["is_success"] for e in episodes),
                "mean_upright_fraction": round(
                    statistics.mean(e["gait"]["mean_upright"] for e in episodes), 4
                ),
                "mean_knee_near_ground_fraction": round(
                    statistics.mean(e["gait"]["mean_knee_near_ground"] for e in episodes), 4
                ),
                "mean_upright_forward_fraction": round(
                    statistics.mean(e["gait"]["upright_forward_fraction"] for e in episodes), 4
                ),
            }
            finished = [e for e in episodes if e["is_success"]]
            row["recent_training"]["finishing_episodes_mean_upright_forward_fraction"] = (
                round(statistics.mean(e["gait"]["upright_forward_fraction"] for e in finished), 4)
                if finished
                else None
            )
        result["arms"][arm] = row
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("window must be positive")
    print(json.dumps(summarize(args.out, args.window), indent=2))
