"""Budgeted diagnostic frames from a verified prefix of a saved trajectory."""

import argparse
import multiprocessing
from pathlib import Path

from PIL import Image

from qwop_lab.artifacts import digest, read_json, sha256, write_json
from qwop_lab.budget import Ledger
from qwop_lab.environment import Case, contract, make_env, reset_case, runtime
from qwop_lab.evaluation import sample
from qwop_lab.replay import annotated, verify_sample


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    data = read_json(args.input)
    assert data["contract"] == contract(runtime())
    assert digest(data["actions"]) == data["actions_sha256"]
    assert digest(data["trace"]) == data["trace_sha256"]
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    lease = Ledger(args.ledger).lease("diagnostic-prefix-gait")
    env = None
    images = []
    try:
        env = make_env(lease)
        obs, info = reset_case(env, Case(data["case"]["seed"], data["case"]["soft_resets"]))
        actual = sample(obs, info)
        offset = actual["raw_time"] - data["initial"]["raw_time"]
        verify_sample(actual, data["initial"], 0, offset)
        stop = min(args.steps, len(data["actions"]))
        assert stop >= 3
        for index, (action, expected) in enumerate(
            zip(data["actions"][:stop], data["trace"][:stop]), 1
        ):
            obs, reward, terminated, truncated, info = env.step(action)
            actual = sample(obs, info, terminated, truncated, reward)
            verify_sample(actual, expected, index, offset)
            if index in (stop // 3, 2 * stop // 3, stop):
                image = Image.fromarray(annotated(env.render(), args.label, actual))
                image.save(output / f"step-{index}.png")
                images.append(image)
        sheet = Image.new("RGB", (320 * len(images), 264), "#eeeeee")
        for i, image in enumerate(images):
            image.thumbnail((320, 264))
            sheet.paste(image, (320 * i, 0))
        sheet.save(output / "samples.png")
        write_json(
            output / "diagnostic.json",
            {
                "source_trace_sha256": sha256(args.input),
                "verified_prefix_steps": lease.used,
                "selection": "Post-hoc gait diagnosis of a slower intermediate checkpoint; no checkpoint or seed selection",
                "label": args.label,
                "contract_id": data["contract"]["contract_id"],
            },
        )
        print(f"Verified and sampled {lease.used} prefix transitions", flush=True)
    finally:
        if env:
            env.close()
        lease.close()
