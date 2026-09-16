"""Bounded architecture experiments with immutable rules and audited proposals.

The researcher submits proposal JSON between trials. This module executes and
measures those proposals; it does not pretend to contain an autonomous LLM.
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path

from .architectures import Architecture
from .artifacts import ROOT, digest, read_json, sha256, write_json
from .budget import Ledger
from .environment import Case, VALIDATION_CASES, contract, runtime
from .evaluation import SCORING, evaluate
from .training import FRESH_PPO_CONFIG, train

ARMS = {
    "baseline": {"fixed": 1048576},
    "search": {"small": 524288, "large": 524288},
    "research": {"proposal-1": 524288, "proposal-2": 524288},
}
CONVENTIONAL_LAYERS = {
    "baseline-fixed": [128, 128],
    "search-small": [64, 64],
    "search-large": [256, 256],
}
CORE_FILES = [
    "agents.py",
    "architectures.py",
    "artifacts.py",
    "browser.py",
    "budget.py",
    "campaign.py",
    "environment.py",
    "evaluation.py",
    "replay.py",
    "training.py",
]
DEVELOPMENT_CASES = [Case(101, 0), Case(101, 1)]


def now():
    return datetime.now(timezone.utc).isoformat()


def protected_sources():
    return {name: sha256(ROOT / "qwop_lab" / name) for name in CORE_FILES}


def init_campaign(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    protocol = {
        "id": "architecture-campaign-v1",
        "created_utc": now(),
        "arms": ARMS,
        "conventional_layers": CONVENTIONAL_LAYERS,
        "training_steps_per_arm": 1048576,
        "evaluation_and_replay_cap_per_arm": 75000,
        "training_wall_seconds_per_arm": 7200,
        "max_parameters": 250000,
        "seed": 42,
        "initialization": "All trials use fresh seeded weights; no historical checkpoint",
        "checkpoint_interval": 262144,
        "ppo_config": FRESH_PPO_CONFIG,
        "training_reward": {"time_cost_mult": 10, "success_reward": 50},
        "contract": contract(runtime()),
        "scoring": SCORING,
        "selection": "Final trial checkpoints only; valid best 100m time, then reliability; "
        "if none finish, mean terminal distance. Intermediate checkpoints diagnose learning.",
        "development_cases": [vars(c) for c in DEVELOPMENT_CASES],
        "final_cases": [vars(c) for c in VALIDATION_CASES],
        "final_case_warning": "Expanded descriptive development evaluation, NOT a held-out "
        "robustness test. Different seeds mostly repeat two physical reset phases.",
        "researcher": {
            "execution": "Codex researcher in the current user task, explicit proposal files",
            "model_snapshot": "Not independently verified by the local runner",
            "paid_model_api_calls_by_runner": 0,
            "scope": "Architecture and deterministic features of the same 60 observations",
            "adaptive_trial": "proposal-2 must cite the completed proposal-1 result hash",
        },
        "comparison_limits": [
            "One training seed and one campaign per method; no statistical superiority claim",
            "Custom architectures are unrestricted within the interface and parameter cap; "
            "ordinary search explores only two predefined standard MLP widths",
            "Equal training interactions and evaluation caps, not equal realized compute cost",
            "Researcher also built the harness and can see baseline results; not blinded",
            "No algorithm sweep or QR-DQN baseline yet; this isolates architecture within PPO",
            "Budget is an initial screen, below upstream's recommended full PPO training",
            "Gait quality is judged from replay, not inferred from speed alone",
            "Before freezing this protocol, a custom-architecture engineering smoke test used "
            "8192 training steps, 30000 evaluation steps and up to 5000 replay steps; "
            "these are disclosed separately and no smoke-test weights are reused",
        ],
        "protected_sources": protected_sources(),
    }
    write_json(folder / "protocol.json", protocol)
    (folder / "protocol.sha256").write_text(sha256(folder / "protocol.json") + "\n")
    for source in sorted((ROOT / "qwop_lab").rglob("*.py")):
        destination = folder / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for arm in ARMS:
        Ledger(folder / arm / "training.sqlite", protocol["training_steps_per_arm"])
        Ledger(folder / arm / "evaluation.sqlite", protocol["evaluation_and_replay_cap_per_arm"])
    definitions = {
        "baseline-fixed": ("baseline", "fixed", [128, 128]),
        "search-small": ("search", "small", [64, 64]),
        "search-large": ("search", "large", [256, 256]),
    }
    for name, (arm, slot, layers) in definitions.items():
        write_json(
            folder / "proposals" / f"{name}.json",
            {
                "arm": arm,
                "slot": slot,
                "architecture": {"layers": layers, "activation": "Tanh"},
                "hypothesis": "Predefined conventional MLP width comparison",
                "expected_signal": "Completion and 100m speed at the final fixed budget",
                "evidence": [],
            },
        )
    return protocol


def verify_protocol(folder):
    folder = Path(folder)
    expected = (folder / "protocol.sha256").read_text().strip()
    if sha256(folder / "protocol.json") != expected:
        raise ValueError("Frozen campaign protocol changed")
    protocol = read_json(folder / "protocol.json")
    if protected_sources() != protocol["protected_sources"]:
        raise ValueError("Protected training/evaluation implementation changed during campaign")
    if contract(runtime()) != protocol["contract"]:
        raise ValueError("Runtime changed during campaign")
    return protocol


def validate_proposal(folder, proposal, protocol):
    arm, slot = proposal["arm"], proposal["slot"]
    if arm not in protocol["arms"] or slot not in protocol["arms"][arm]:
        raise ValueError("Proposal is outside the campaign's fixed trial allocation")
    architecture = Architecture.from_dict(proposal["architecture"])
    if (
        not proposal.get("hypothesis", "").strip()
        or not proposal.get("expected_signal", "").strip()
    ):
        raise ValueError("A proposal requires a hypothesis and a predicted observable result")
    if arm in ("baseline", "search"):
        name = f"{arm}-{slot}"
        expected = Architecture(tuple(protocol["conventional_layers"][name]))
        if architecture != expected:
            raise ValueError("Conventional candidates are fixed before training")
    for evidence in proposal.get("evidence", []):
        path = (Path(folder) / evidence["path"]).resolve()
        if not path.is_relative_to(Path(folder).resolve()):
            raise ValueError("Evidence must be a campaign artifact")
        if sha256(path) != evidence["sha256"]:
            raise ValueError("Proposal evidence changed")
    if arm == "research" and slot == "proposal-2":
        prior = Path(folder) / "research" / "proposal-1" / "result.json"
        if read_json(prior)["status"] != "complete":
            raise ValueError("The first research trial must finish before the adaptive proposal")
        if not any(
            e["path"] == "research/proposal-1/result.json" and e["sha256"] == sha256(prior)
            for e in proposal.get("evidence", [])
        ):
            raise ValueError("Adaptive proposal must cite the first trial's observed result")
    return architecture


def run_trial(folder, proposal_file):
    folder = Path(folder).resolve()
    protocol = verify_protocol(folder)
    proposal = read_json(proposal_file)
    architecture = validate_proposal(folder, proposal, protocol)
    arm, slot = proposal["arm"], proposal["slot"]
    steps = protocol["arms"][arm][slot]
    output = folder / arm / slot
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "proposal.json", proposal)
    identity = architecture.identity()
    if identity.get("source_file"):
        shutil.copy2(identity["source_file"], output / "architecture-source.py")
    result = {
        "status": "running",
        "started_utc": now(),
        "arm": arm,
        "slot": slot,
        "protocol_sha256": sha256(folder / "protocol.json"),
        "proposal_sha256": sha256(output / "proposal.json"),
        "architecture": identity,
        "requested_steps": steps,
        "checkpoints": [],
    }
    write_json(output / "result.json", result)
    training_ledger = Ledger(folder / arm / "training.sqlite")
    evaluation_ledger = Ledger(folder / arm / "evaluation.sqlite")
    try:
        checkpoint = train(
            None,
            output / "training",
            training_ledger,
            steps=steps,
            seed=protocol["seed"],
            architecture=architecture,
            config=protocol["ppo_config"],
            checkpoint_interval=protocol["checkpoint_interval"],
            max_parameters=protocol["max_parameters"],
            max_seconds=protocol["training_wall_seconds_per_arm"]
            * steps
            / protocol["training_steps_per_arm"],
        )
        verify_protocol(folder)
        for step in range(
            protocol["checkpoint_interval"], steps + 1, protocol["checkpoint_interval"]
        ):
            path = output / "training" / "checkpoints" / f"step-{step:09d}.zip"
            if step == steps:
                path = checkpoint
            summary = evaluate(
                path,
                DEVELOPMENT_CASES,
                output / f"eval-{step:09d}",
                evaluation_ledger,
                f"{arm}/{slot}@{step}",
            )
            result["checkpoints"].append(
                {"steps": step, "summary": summary, "checkpoint_sha256": sha256(path)}
            )
            write_json(output / "result.json", result)
        result.update(
            status="complete",
            final_checkpoint=str(checkpoint),
            final_checkpoint_sha256=sha256(checkpoint),
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


def selection_key(summary):
    best = summary["best_100m_game_seconds"]
    if best is not None:
        return (0, best, -summary["valid_100m_finish_rate"], -summary["mean_distance"])
    return (1, 0, 0, -summary["mean_distance"])


def campaign_status(folder):
    folder = Path(folder)
    protocol = read_json(folder / "protocol.json")
    arms = {}
    for arm, slots in protocol["arms"].items():
        trials = {}
        for slot in slots:
            path = folder / arm / slot / "result.json"
            if path.exists():
                result = read_json(path)
                progress = folder / arm / slot / "training" / "progress.json"
                trials[slot] = {k: result[k] for k in ("status", "requested_steps")}
                if result.get("final_summary"):
                    trials[slot]["summary"] = result["final_summary"]
                if progress.exists():
                    trials[slot]["progress"] = read_json(progress)
            else:
                trials[slot] = {"status": "pending"}
        arms[arm] = {
            "trials": trials,
            "training_budget": Ledger(folder / arm / "training.sqlite").status(),
            "evaluation_budget": Ledger(folder / arm / "evaluation.sqlite").status(),
        }
    return {"protocol_sha256": sha256(folder / "protocol.json"), "arms": arms}


def finish_campaign(folder):
    from .replay import replay

    folder = Path(folder).resolve()
    protocol = verify_protocol(folder)
    if (folder / "comparison.json").exists():
        raise FileExistsError("Campaign already finalized")
    selected = {}
    # Validate every completed arm before spending any final-evaluation budget.
    for arm, slots in protocol["arms"].items():
        results = [read_json(folder / arm / slot / "result.json") for slot in slots]
        if any(r["status"] != "complete" for r in results):
            raise ValueError(f"Unfinished arm: {arm}")
        if sum(r["requested_steps"] for r in results) != protocol["training_steps_per_arm"]:
            raise ValueError("Unequal training allocation")
        for r in results:
            if r["protocol_sha256"] != sha256(folder / "protocol.json"):
                raise ValueError("Trial used a different protocol")
            if sha256(r["final_checkpoint"]) != r["final_checkpoint_sha256"]:
                raise ValueError("Final checkpoint changed")
            if r["checkpoints"][-1]["checkpoint_sha256"] != r["final_checkpoint_sha256"]:
                raise ValueError("Selected score was measured on different weights")
            training = read_json(folder / arm / r["slot"] / "training" / "manifest.json")
            if training["status"] != "complete" or training["actual_steps"] != r["requested_steps"]:
                raise ValueError("Training did not consume its allocated steps")
        selected[arm] = min(results, key=lambda r: selection_key(r["final_summary"]))
    write_json(
        folder / "selection.json",
        {
            "created_utc": now(),
            "rule": protocol["selection"],
            "selected": {
                arm: {"slot": r["slot"], "checkpoint_sha256": r["final_checkpoint_sha256"]}
                for arm, r in selected.items()
            },
        },
    )
    comparisons = {}
    for arm, result in selected.items():
        ledger = Ledger(folder / arm / "evaluation.sqlite")
        eval_dir = folder / arm / "final-eval"
        if eval_dir.exists():
            evaluation = read_json(eval_dir / "manifest.json")
            if (
                evaluation["status"] != "complete"
                or evaluation["model"]["sha256"] != result["final_checkpoint_sha256"]
                or evaluation["contract"] != protocol["contract"]
            ):
                raise ValueError("Existing final evaluation is incomplete or incompatible")
            # Verify saved trajectories before reusing a completed final evaluation.
            from .evaluation import summarize

            episodes = []
            for case in VALIDATION_CASES:
                episode = read_json(eval_dir / f"{case.id}.json")
                if digest(episode["trace"]) != episode["trace_sha256"]:
                    raise ValueError("Saved final trajectory changed")
                episodes.append(episode)
            summary = summarize(episodes)
        else:
            summary = evaluate(
                result["final_checkpoint"], VALIDATION_CASES, eval_dir, ledger, f"selected-{arm}"
            )
        # Same predetermined start case in every video, independent of best-case score.
        video_path = folder / arm / "replay.mp4"
        if video_path.exists():
            video = read_json(video_path.with_suffix(".json"))
            if not video["verified"] or video["video_sha256"] != sha256(video_path):
                raise ValueError("Existing replay changed or was not verified")
        else:
            video = replay(
                eval_dir / "seed-101-soft-0.json", video_path, ledger, f"{arm}: {result['slot']}"
            )
        comparisons[arm] = {
            "selected_slot": result["slot"],
            "summary": summary,
            "video": video,
            "trainable_parameters": read_json(
                folder / arm / result["slot"] / "training" / "manifest.json"
            )["trainable_parameters"],
        }
    report = {
        "status": "complete",
        "created_utc": now(),
        "protocol": protocol,
        "comparisons": comparisons,
        "budget": campaign_status(folder),
        "result_digest": digest(comparisons),
    }
    write_json(folder / "comparison.json", report)
    return report
