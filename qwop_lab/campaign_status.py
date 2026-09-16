"""Read-only status view including explicitly amended recovery attempts."""

from pathlib import Path

from .artifacts import read_json, sha256
from .campaign import campaign_status


def status(folder):
    folder = Path(folder)
    result = campaign_status(folder)
    amendment = folder / "amendment-001.json"
    if amendment.exists():
        result["amendment"] = {**read_json(amendment), "sha256": sha256(amendment)}
        for trial in (folder / "research").glob("*/result.json"):
            value = read_json(trial)
            slot = value["slot"]
            entry = {k: value[k] for k in ("status", "requested_steps")}
            if value.get("final_summary"):
                entry["summary"] = value["final_summary"]
            if value.get("error"):
                entry["error"] = value["error"]
            progress = trial.parent / "training/progress.json"
            if progress.exists():
                entry["progress"] = read_json(progress)
            result["arms"]["research"]["trials"][slot] = entry
    final = folder / "comparison.json"
    result["status"] = read_json(final)["status"] if final.exists() else "in_progress"
    return result
