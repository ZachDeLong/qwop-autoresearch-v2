"""Read progress snapshots; atomic writers retry brief Windows reader locks."""

import argparse
import json
from pathlib import Path

from qwop_lab.artifacts import read_json


read_shared_json = read_json


def status(folder):
    folder = Path(folder)
    protocol = read_shared_json(folder / "protocol.json")
    rows = []
    for seed in protocol["seeds"]:
        for design in protocol["designs"]:
            path = folder / design / f"seed-{seed}"
            result = (
                read_shared_json(path / "result.json") if (path / "result.json").exists() else {}
            )
            progress = path / "training/progress.json"
            p = read_shared_json(progress) if progress.exists() else {}
            state = result.get("status", "pending")
            if (path / "recovery-training/manifest.json").exists():
                continued = read_shared_json(path / "recovery-training/manifest.json")
                progress = path / "recovery-training/progress.json"
                p = read_shared_json(progress) if progress.exists() else {}
                if p:
                    p["steps_per_second"] = (p["steps"] - continued["prior_spent_steps"]) / p[
                        "wall_seconds"
                    ]
                state = "continuing" if continued["status"] == "running" else "continued-evaluation"
                if (path / "recovered-result.json").exists():
                    result = read_shared_json(path / "recovered-result.json")
                    state = (
                        "complete-resumed" if result["status"] == "complete" else result["status"]
                    )
            rows.append(
                {
                    "design": design,
                    "seed": seed,
                    "status": state,
                    "steps": p.get("steps"),
                    "steps_per_second": round(p.get("steps_per_second", 0)),
                    "checkpoints": len(list(path.glob("eval-*/summary.json"))),
                    "error": result.get("error"),
                }
            )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    rows = status(args.campaign)
    if args.compact:
        for row in rows:
            print(
                f"{row['design']:10} {row['seed']:2} {row['status']:20} {row['steps'] or 0:>9,} steps   {row['steps_per_second']} steps/s   {row['checkpoints']} evaluations"
            )
    else:
        print(json.dumps(rows, indent=2))
