"""Finish-line speed metrics plus verifiable per-step trajectory artifacts."""

import statistics
import time
from pathlib import Path

import numpy as np

from .agents import load_policy
from .artifacts import digest, observation_hash, provenance, read_json, sha256, write_json
from .environment import case_dict, contract, make_env, reset_case, runtime

SCORING = {
    "id": "finish-line-v2",
    "primary": "best_100m_game_seconds",
    "direction": "minimize",
    "valid_finish": "Recorded 100m crossing and native terminal success",
    "clock": "info.time = CORE.game.scoreTime / 10; unrounded game display seconds",
    "crossing": "First observed torso distance >=100m; four-frame sampling, no interpolation",
    "reliability": "Report completion rate separately; no implicit 100% threshold",
    "verification": "A record candidate requires successful deterministic trajectory replay",
}


def finish_seconds(episode):
    crossing = episode["first_100m_score_time"]
    return crossing / 10 if episode["is_success"] and crossing is not None else None


def sample(obs, info, terminated=False, truncated=False, reward=None):
    if not np.isfinite(obs).all():
        raise ValueError("Nonfinite observation")
    return {
        "obs_sha256": observation_hash(obs),
        "raw_time": float(info["time"]),
        "score_time": float(info["time"]) * 10,
        "distance": float(info["distance"]),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "is_success": bool(info["is_success"]),
        "reward": None if reward is None else float(reward),
    }


def rollout(env, policy, case):
    start = time.perf_counter()
    obs, info = reset_case(env, case)
    initial = sample(obs, info)
    actions, trace = [], []
    first_100m = None
    while True:
        action = policy(obs)
        if not isinstance(action, int) or not 0 <= action < 16:
            raise ValueError(f"Policy returned invalid action: {action!r}")
        obs, reward, terminated, truncated, info = env.step(action)
        row = sample(obs, info, terminated, truncated, reward)
        actions.append(action)
        trace.append(row)
        if first_100m is None and row["distance"] >= 100:
            first_100m = row["score_time"]
        if terminated or truncated:
            break
    final = trace[-1]
    return {
        "case": case_dict(case),
        "initial": initial,
        "actions": actions,
        "trace": trace,
        "actions_sha256": digest(actions),
        "trace_sha256": digest(trace),
        "num_steps": len(actions),
        "physics_frames_requested": len(actions) * 4,
        "is_success": final["is_success"] and final["terminated"],
        "distance": final["distance"],
        "score_time": final["score_time"],
        "first_100m_score_time": first_100m,
        "wall_seconds_including_reset": time.perf_counter() - start,
    }


def summarize(episodes):
    if not episodes:
        raise ValueError("An evaluation must contain at least one episode")
    finished = [e for e in episodes if e["is_success"]]
    times = [e["score_time"] for e in finished]
    crossing_times = [t for e in episodes if (t := finish_seconds(e)) is not None]
    return {
        "scoring": SCORING,
        "valid_100m_finishes": len(crossing_times),
        "valid_100m_finish_rate": len(crossing_times) / len(episodes),
        "best_100m_game_seconds": min(crossing_times) if crossing_times else None,
        "mean_100m_game_seconds": statistics.mean(crossing_times) if crossing_times else None,
        "median_100m_game_seconds": statistics.median(crossing_times) if crossing_times else None,
        "legacy_finish_time_units": "Native terminal info.time * 10; not game display seconds",
        "episodes": len(episodes),
        "finishes": len(finished),
        "success_rate": len(finished) / len(episodes),
        "mean_finish_score_time": statistics.mean(times) if times else None,
        "median_finish_score_time": statistics.median(times) if times else None,
        "best_finish_score_time": min(times) if times else None,
        "mean_distance": statistics.mean(e["distance"] for e in episodes),
        "total_steps": sum(e["num_steps"] for e in episodes),
        "unique_initial_observations": len({e["initial"]["obs_sha256"] for e in episodes}),
        "unique_action_sequences": len({e["actions_sha256"] for e in episodes}),
        "unique_trajectories": len({e["trace_sha256"] for e in episodes}),
        "interpretation": "Descriptive validation cases, not independent training replications",
    }


def select_replay(episodes):
    finished = [e for e in episodes if finish_seconds(e) is not None]
    return (
        min(finished, key=finish_seconds)
        if finished
        else max(episodes, key=lambda e: e["distance"])
    )


def rescore(folder, output):
    """Derive a new score from saved traces without overwriting original results."""
    folder, output = Path(folder), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = read_json(folder / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Only complete evaluations can be rescored")
    episodes, inputs = [], {}
    for case in manifest["cases"]:
        path = folder / f"{case['id']}.json"
        episode = read_json(path)
        if episode["contract"] != manifest["contract"]:
            raise ValueError("Episode and evaluation contracts differ")
        if digest(episode["trace"]) != episode["trace_sha256"]:
            raise ValueError("Recorded trace hash mismatch")
        crossing = next((r["score_time"] for r in episode["trace"] if r["distance"] >= 100), None)
        if crossing != episode["first_100m_score_time"]:
            raise ValueError("Recorded crossing disagrees with trajectory")
        episodes.append(episode)
        inputs[path.name] = sha256(path)
    result = {
        "summary": summarize(episodes),
        "model": manifest["model"],
        "contract": manifest["contract"],
        "source_manifest_sha256": sha256(folder / "manifest.json"),
        "source_episode_hashes": inputs,
        "selected_case": select_replay(episodes)["case"],
        "provenance": provenance(),
        "additional_environment_steps": 0,
    }
    write_json(output, result)
    return result


def evaluate(checkpoint, cases, out_dir, ledger, label):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    policy, model_id = load_policy(checkpoint)
    meta = {
        "schema_version": 2,
        "scoring": SCORING,
        "label": label,
        "model": model_id,
        "contract": contract(runtime()),
        "provenance": provenance(),
        "cases": [case_dict(c) for c in cases],
        "status": "running",
    }
    write_json(out_dir / "manifest.json", meta)
    episodes = []
    lease = ledger.lease(label)
    env = None
    try:
        env = make_env(lease)
        for case in cases:
            episode = rollout(env, policy, case)
            episodes.append(episode)
            write_json(out_dir / f"{case.id}.json", {"contract": meta["contract"], **episode})
            crossing = finish_seconds(episode)
            crossing_label = (
                f"{crossing:.3f}s to 100m" if crossing is not None else "no valid finish"
            )
            outcome = (
                "finish"
                if episode["is_success"]
                else ("timeout" if episode["trace"][-1]["truncated"] else "fall")
            )
            print(
                f"{label} {case.id}: {outcome} {episode['distance']:.2f}m / {crossing_label}",
                flush=True,
            )
        summary = summarize(episodes)
        write_json(out_dir / "summary.json", summary)
        best = select_replay(episodes)
        write_json(out_dir / "selected-replay.json", {"contract": meta["contract"], **best})
        meta.update(status="complete", summary=summary)
        return summary
    except BaseException as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        try:
            if env:
                env.close()
        finally:
            lease.close()
            meta["steps_used"] = lease.used
            meta["budget"] = ledger.status()
            write_json(out_dir / "manifest.json", meta)
