"""Dispatch the single recorded continuation when the batch has a free slot."""

import argparse
from pathlib import Path
import subprocess
import sys
import time

from read_replication_status import read_shared_json
from qwop_lab.artifacts import ROOT, sha256, write_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    args = parser.parse_args()
    folder = Path(args.campaign).resolve()
    deadline = time.monotonic() + 7200
    waiting_for = ("baseline/seed-17", "kinematic/seed-17", "kinematic/seed-29")
    print("Waiting for the first three uninterrupted runs to finish", flush=True)
    while True:
        states = []
        for relative in waiting_for:
            result = folder / relative / "result.json"
            states.append(read_shared_json(result).get("status") if result.exists() else "pending")
        if "failed" in states:
            raise RuntimeError("Another training run failed; continuation dispatch needs review")
        if (
            all(state == "complete" for state in states)
            and len(list(folder.glob("*/seed-*/running.lock"))) < 3
        ):
            break
        if time.monotonic() > deadline:
            raise TimeoutError("No continuation slot became available within two hours")
        time.sleep(10)
    write_json(
        folder / "continuation-dispatch.json",
        {
            "implementation_sha256": sha256(__file__),
            "amendment_sha256": sha256(folder / "resume-amendment.json"),
            "active_original_workers": len(list(folder.glob("*/seed-*/running.lock"))),
        },
    )
    print("Starting the recorded continuation within the existing allowance", flush=True)
    with (folder / "continuation.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "qwop_lab.replication_resume",
                "resume",
                "--out",
                str(folder),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("Continuation and its evaluations completed", flush=True)
