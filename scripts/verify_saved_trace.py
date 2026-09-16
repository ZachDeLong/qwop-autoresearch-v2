"""Verify saved body-state transitions without rendering a video."""

import argparse
import multiprocessing

from qwop_lab.artifacts import digest, read_json, sha256, write_json
from qwop_lab.budget import Ledger
from qwop_lab.environment import Case, contract, make_env, reset_case, runtime
from qwop_lab.evaluation import sample
from qwop_lab.replay import verify_sample


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = read_json(args.input)
    assert data["contract"] == contract(runtime())
    assert digest(data["actions"]) == data["actions_sha256"]
    assert digest(data["trace"]) == data["trace_sha256"]
    ledger = Ledger(args.ledger, len(data["actions"]))
    lease = ledger.lease("saved-trajectory-check")
    env = None
    try:
        env = make_env(lease)
        obs, info = reset_case(env, Case(data["case"]["seed"], data["case"]["soft_resets"]))
        actual = sample(obs, info)
        offset = actual["raw_time"] - data["initial"]["raw_time"]
        verify_sample(actual, data["initial"], 0, offset)
        for index, (action, expected) in enumerate(zip(data["actions"], data["trace"]), 1):
            obs, reward, terminated, truncated, info = env.step(action)
            actual = sample(obs, info, terminated, truncated, reward)
            verify_sample(actual, expected, index, offset)
        write_json(
            args.out,
            {
                "verified_transitions": lease.used,
                "source_trace_sha256": sha256(args.input),
                "contract_id": data["contract"]["contract_id"],
                "reset_raw_time_offset": offset,
            },
        )
        print(f"Verified {lease.used} transitions", flush=True)
    finally:
        if env:
            env.close()
        lease.close()
