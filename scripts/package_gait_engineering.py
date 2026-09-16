"""Archive the disclosed setup diagnostics and both engineering smoke attempts."""

import argparse
from pathlib import Path
import tarfile

from qwop_lab.artifacts import ROOT, sha256, write_json
from qwop_lab.budget import Ledger

RUNS = {
    "gait-diagnostic-001": "Browser startup failed on file:// access before any game steps.",
    "gait-diagnostic-002": "Historical speed-policy geometry check and verified replay.",
    "gait-smoke-001": "Training/evaluation succeeded; both long replays rejected by the old clock bound.",
    "gait-smoke-002": "Complete smoke pair, including full timeout replays and independent artifact audit.",
}


def package(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    evidence, files = {}, []
    for run, note in RUNS.items():
        folder = ROOT / "runs" / run
        ledgers = {}
        for path in sorted(folder.rglob("*.sqlite")):
            state = Ledger(path).status()
            if any(not lease["closed"] for lease in state["leases"]):
                raise ValueError("Cannot package an active diagnostic")
            ledgers[str(path.relative_to(folder))] = state
        evidence[run] = {
            "description": note,
            "ledgers": ledgers,
            "charged_steps": sum(s["charged_steps"] for s in ledgers.values()),
            "confirmed_steps": sum(s["confirmed_steps"] for s in ledgers.values()),
        }
        files.extend(p for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix != ".mp4")
    total = sum(e["charged_steps"] for e in evidence.values())
    write_json(
        destination / "costs.json",
        {
            "runs": evidence,
            "total_charged_steps": total,
            "included_in_main_study_budget": False,
            "weights_reused_in_main_study": False,
        },
    )
    write_json(
        destination / "artifact-index.json",
        {str(p.relative_to(ROOT / "runs")): sha256(p) for p in files},
    )
    with tarfile.open(destination / "artifacts.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT / "runs")))
    rows = [
        "# Gait-study engineering evidence",
        "",
        "These diagnostics preceded the main training runs. "
        "No weights were reused. All failed attempts remain in the archive; videos and proprietary game assets are excluded.",
        "",
        "| Attempt | Charged steps | Outcome |",
        "| --- | ---: | --- |",
    ]
    rows += [
        f"| {run} | {item['charged_steps']:,} | {item['description']} |"
        for run, item in evidence.items()
    ]
    rows += [
        "",
        f"Total additional engineering cost: **{total:,} environment steps**.",
        "",
        "The first smoke's physical states matched through the failed checks. The fixed clock check "
        "was additionally validated against 5,000 saved transitions with no physical-state tolerance, "
        "then exercised through both complete timeout replays in smoke-002. Its audit also recomputed "
        "all raw-pose gait metrics, verified identical paired initial policy weights, and reconciled the budgets.",
        "",
        "The historical diagnostic finishes both reset phases. Across those episodes its mean pelvis "
        "height is about 0.406 m, knees are near the ground for about 98% of samples, and less than "
        "0.04% of positive forward displacement qualifies as upright under the exploratory thresholds.",
        "",
    ]
    (destination / "report.md").write_text("\n".join(rows))
    print(f"Archived {len(files)} artifacts; {total:,} additional environment steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    package(parser.parse_args().out)
