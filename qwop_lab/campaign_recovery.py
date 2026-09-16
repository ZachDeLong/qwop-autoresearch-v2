"""Explicit, hash-linked amendment for a failed browser transport in campaign v1.

Original protocol, failed result, and ledger charges remain untouched. A fresh
retry spends only the failed slot's unused allowance, rounded down to a complete
PPO rollout. This produces an amended exploratory comparison, not an unamended run.
"""

from pathlib import Path
import shutil

from .architectures import Architecture
from .artifacts import read_json, sha256, write_json
from .budget import Ledger
from .campaign import DEVELOPMENT_CASES, campaign_status, now, selection_key, verify_protocol
from .environment import VALIDATION_CASES
from .evaluation import evaluate
from .replay import replay
from .training import train


def amend(folder):
    folder = Path(folder).resolve()
    protocol = verify_protocol(folder)
    amendment = folder / "amendment-001.json"
    if amendment.exists():
        raise FileExistsError(amendment)
    failed_path = folder / "research/proposal-1/result.json"
    failed = read_json(failed_path)
    if failed["status"] != "failed" or "Ambiguous game response" not in failed.get("error", ""):
        raise ValueError(
            "This amendment applies only to the recorded first-trial transport failure"
        )
    meta = read_json(folder / "research/proposal-1/training/manifest.json")
    spent = meta["actual_steps"]
    rollout = protocol["ppo_config"]["n_steps"]
    remaining = (protocol["arms"]["research"]["proposal-1"] - spent) // rollout * rollout
    if remaining <= 0:
        raise ValueError("No complete rollout remains in the original slot allowance")
    proposal = read_json(folder / "research/proposal-1/proposal.json")
    value = {
        "id": "transport-recovery-001",
        "created_utc": now(),
        "protocol_sha256": sha256(folder / "protocol.json"),
        "failed_result_sha256": sha256(failed_path),
        "failed_training_manifest_sha256": sha256(
            folder / "research/proposal-1/training/manifest.json"
        ),
        "implementation_sha256": sha256(Path(__file__)),
        "diagnosis": "Browser response timed out; exact cause not established. No action was resent.",
        "failed_attempt_charged_steps": spent,
        "retry_slot": "proposal-1-retry",
        "retry_steps": remaining,
        "unused_rounding_steps": protocol["arms"]["research"]["proposal-1"] - spent - remaining,
        "retry_architecture": proposal["architecture"],
        "changes": [
            "Allow one fresh retry of the identical first architecture within that slot's unused budget",
            "Evaluate the retry only at its final trained checkpoint because its shortened allocation no longer divides the original checkpoint interval",
            "Second proposal must cite both the original failed result and the completed retry",
            "The original training/evaluation ledgers and total caps stay in force; no additional training steps are granted",
            "Compare the completed retry as trial one, while reporting its shorter effective training and failed-attempt cost",
            "If another transport error occurs, preserve it and leave this campaign incomplete",
        ],
    }
    write_json(amendment, value)
    (folder / "amendment-001.sha256").write_text(sha256(amendment) + "\n")
    shutil.copy2(__file__, folder / "source/qwop_lab/campaign_recovery.py")
    return value


def verified_amendment(folder):
    folder = Path(folder)
    protocol = verify_protocol(folder)
    path = folder / "amendment-001.json"
    if sha256(path) != (folder / "amendment-001.sha256").read_text().strip():
        raise ValueError("Recovery amendment changed")
    amendment = read_json(path)
    if amendment["protocol_sha256"] != sha256(folder / "protocol.json"):
        raise ValueError("Recovery amendment belongs to another protocol")
    if amendment["implementation_sha256"] != sha256(Path(__file__)):
        raise ValueError("Recovery implementation changed")
    if amendment["failed_result_sha256"] != sha256(folder / "research/proposal-1/result.json"):
        raise ValueError("Original failed result changed")
    return protocol, amendment


def run_research(folder, proposal_file=None):
    folder = Path(folder).resolve()
    protocol, amendment = verified_amendment(folder)
    if proposal_file is None:
        proposal = read_json(folder / "research/proposal-1/proposal.json")
        proposal["slot"] = amendment["retry_slot"]
        proposal["hypothesis"] += (
            " Identical architecture retried after the recorded transport failure."
        )
        steps = amendment["retry_steps"]
        interval = steps
    else:
        proposal = read_json(proposal_file)
        if proposal["arm"] != "research" or proposal["slot"] != "proposal-2":
            raise ValueError("Expected the second research proposal")
        if not proposal.get("hypothesis") or not proposal.get("expected_signal"):
            raise ValueError("Missing research hypothesis")
        for relative in (
            "research/proposal-1/result.json",
            "research/proposal-1-retry/result.json",
        ):
            path = folder / relative
            if (
                relative.endswith("proposal-1-retry/result.json")
                and read_json(path)["status"] != "complete"
            ):
                raise ValueError("Retry must complete before adaptive proposal")
            if not any(
                e["path"] == relative and e["sha256"] == sha256(path)
                for e in proposal.get("evidence", [])
            ):
                raise ValueError(
                    "Adaptive proposal must cite both the failed and completed prior trial"
                )
        for evidence in proposal.get("evidence", []):
            path = (folder / evidence["path"]).resolve()
            if not path.is_relative_to(folder) or sha256(path) != evidence["sha256"]:
                raise ValueError("Invalid adaptive evidence")
        steps = protocol["arms"]["research"]["proposal-2"]
        interval = protocol["checkpoint_interval"]
    design = Architecture.from_dict(proposal["architecture"])
    output = folder / "research" / proposal["slot"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "proposal.json", proposal)
    if design.identity().get("source_file"):
        shutil.copy2(design.identity()["source_file"], output / "architecture-source.py")
    result = {
        "status": "running",
        "arm": "research",
        "slot": proposal["slot"],
        "started_utc": now(),
        "requested_steps": steps,
        "architecture": design.identity(),
        "protocol_sha256": sha256(folder / "protocol.json"),
        "amendment_sha256": sha256(folder / "amendment-001.json"),
        "proposal_sha256": sha256(output / "proposal.json"),
        "checkpoints": [],
    }
    write_json(output / "result.json", result)
    training_ledger = Ledger(folder / "research/training.sqlite")
    evaluation_ledger = Ledger(folder / "research/evaluation.sqlite")
    try:
        model = train(
            None,
            output / "training",
            training_ledger,
            steps=steps,
            seed=protocol["seed"],
            architecture=design,
            config=protocol["ppo_config"],
            checkpoint_interval=interval,
            max_parameters=protocol["max_parameters"],
            max_seconds=protocol["training_wall_seconds_per_arm"]
            * steps
            / protocol["training_steps_per_arm"],
        )
        verified_amendment(folder)
        for step in range(interval, steps + 1, interval):
            checkpoint = (
                model if step == steps else output / "training/checkpoints" / f"step-{step:09d}.zip"
            )
            summary = evaluate(
                checkpoint,
                DEVELOPMENT_CASES,
                output / f"eval-{step:09d}",
                evaluation_ledger,
                f"research/{proposal['slot']}@{step}",
            )
            result["checkpoints"].append(
                {"steps": step, "summary": summary, "checkpoint_sha256": sha256(checkpoint)}
            )
        result.update(
            status="complete",
            final_checkpoint=str(model),
            final_checkpoint_sha256=sha256(model),
            final_summary=result["checkpoints"][-1]["summary"],
        )
        return result
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result.update(
            finished_utc=now(),
            training_budget=training_ledger.status(),
            evaluation_budget=evaluation_ledger.status(),
        )
        write_json(output / "result.json", result)


def finish(folder):
    folder = Path(folder).resolve()
    protocol, amendment = verified_amendment(folder)
    if (folder / "comparison.json").exists():
        raise FileExistsError("Already finalized")
    effective_arms = {
        **protocol["arms"],
        "research": {
            amendment["retry_slot"]: amendment["retry_steps"],
            "proposal-2": 524288,
        },
    }
    choices = {}
    for arm, slots in effective_arms.items():
        candidates = []
        for slot, steps in slots.items():
            path = folder / arm / slot
            result = read_json(path / "result.json")
            meta = read_json(path / "training/manifest.json")
            if (
                result["status"] != "complete"
                or meta["status"] != "complete"
                or meta["actual_steps"] != steps
            ):
                raise ValueError(f"Incomplete trial: {arm}/{slot}")
            if result["protocol_sha256"] != sha256(folder / "protocol.json"):
                raise ValueError("Trial protocol mismatch")
            if sha256(result["final_checkpoint"]) != result["checkpoints"][-1]["checkpoint_sha256"]:
                raise ValueError("Selected score belongs to different weights")
            candidates.append(result)
        choices[arm] = min(candidates, key=lambda r: selection_key(r["final_summary"]))
    write_json(
        folder / "selection.json",
        {
            "created_utc": now(),
            "amendment_sha256": sha256(folder / "amendment-001.json"),
            "selected": {arm: r["slot"] for arm, r in choices.items()},
        },
    )
    comparisons = {}
    for arm, result in choices.items():
        ledger = Ledger(folder / arm / "evaluation.sqlite")
        eval_dir = folder / arm / "final-eval"
        summary = evaluate(
            result["final_checkpoint"], VALIDATION_CASES, eval_dir, ledger, f"selected-{arm}"
        )
        video = replay(
            eval_dir / "seed-101-soft-0.json",
            folder / arm / "replay.mp4",
            ledger,
            f"{arm}: {result['slot']}",
        )
        meta = read_json(folder / arm / result["slot"] / "training/manifest.json")
        comparisons[arm] = {
            "selected_slot": result["slot"],
            "summary": summary,
            "video": video,
            "trainable_parameters": meta["trainable_parameters"],
        }
    report = {
        "status": "complete_with_recovery_amendment",
        "created_utc": now(),
        "protocol": protocol,
        "amendment": amendment,
        "effective_arms": effective_arms,
        "comparisons": comparisons,
        "budget": campaign_status(folder),
    }
    write_json(folder / "comparison.json", report)
    return report


def main():
    import argparse
    import json
    import multiprocessing

    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["amend", "retry", "adaptive", "finish"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--proposal")
    args = parser.parse_args()
    if args.action == "adaptive":
        if not args.proposal:
            parser.error("adaptive requires --proposal")
        result = run_research(args.out, args.proposal)
    else:
        result = {"amend": amend, "retry": run_research, "finish": finish}[args.action](args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
