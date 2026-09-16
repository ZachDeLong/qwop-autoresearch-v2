"""Bounded browser startup diagnostic; no policy training."""

import multiprocessing
import json
import argparse

from qwop_lab import environment
from qwop_lab.budget import Ledger
from qwop_lab.browser import LabServer


class DiagnosticServer(LabServer):
    async def _launch_browser(self):
        try:
            await super()._launch_browser()
        finally:
            if self._driver:
                print("BROWSER URL:", self._driver.current_url, flush=True)
                print("BROWSER ERRORS:", self._driver.get_log("browser"), flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--steps", type=int, choices=range(17), default=16)
    args = parser.parse_args()
    environment.LabServer = DiagnosticServer
    ledger = Ledger(args.ledger, max(1, args.steps))
    lease = ledger.lease("startup-check")
    env = None
    try:
        env = environment.make_env(lease)
        obs, info = environment.reset_case(env, environment.Case(101, 0))
        for _ in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break
        print(json.dumps({"shape": list(obs.shape), "info": info}, default=float), flush=True)
    finally:
        if env:
            env.close()
        lease.close()
        print(json.dumps(ledger.status()), flush=True)
