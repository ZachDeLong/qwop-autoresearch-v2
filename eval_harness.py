"""
eval_harness.py — FROZEN evaluation harness for QWOP autoresearch v2.

DO NOT MODIFY. This script evaluates any trained agent on 100 episodes
and produces standardized results for baseline vs Claude comparison.

The agent must implement one of two interfaces:
  1. get_action(obs: np.ndarray) -> int          (policy-based)
  2. get_action_sequence() -> list[int]           (replay-based)

Usage:
  python eval_harness.py --agent claude/agent.py --out results/eval_claude.json
  python eval_harness.py --agent baseline/agent.py --out results/eval_baseline.json
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import gymnasium as gym
import numpy as np
import qwop_gym  # noqa: F401

# === FROZEN CONSTANTS ===
NUM_EVAL_EPISODES = 100
FRAMES_PER_STEP = 4
MAX_EPISODE_STEPS = 5000
GAME_TIME_MULTIPLIER = 10  # qwop-gym reports scoreTime/10

# Chrome/driver paths — override via env vars for non-Windows systems
CHROME_PATH = os.environ.get(
    "QWOP_CHROME_PATH",
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
)
CHROMEDRIVER_PATH = os.environ.get(
    "QWOP_CHROMEDRIVER_PATH",
    os.path.expanduser("~/.cache/selenium/chromedriver/win64/145.0.7632.117/chromedriver.exe"),
)


def load_agent(agent_path: str):
    """Load an agent module and return (get_action_fn, get_sequence_fn).

    Exactly one of the two will be non-None.
    """
    spec = importlib.util.spec_from_file_location("agent_module", agent_path)
    module = importlib.util.module_from_spec(spec)

    # Allow the agent to import from its own directory
    agent_dir = os.path.dirname(os.path.abspath(agent_path))
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)

    spec.loader.exec_module(module)

    has_action = hasattr(module, "get_action") and callable(module.get_action)
    has_sequence = hasattr(module, "get_action_sequence") and callable(module.get_action_sequence)

    if not has_action and not has_sequence:
        raise ValueError(
            f"Agent at {agent_path} must implement get_action(obs) -> int "
            f"or get_action_sequence() -> list[int]"
        )

    if has_action and has_sequence:
        print("Warning: agent implements both interfaces; using get_action (policy-based)")

    get_action_fn = module.get_action if has_action else None
    get_sequence_fn = module.get_action_sequence if not has_action and has_sequence else None

    return get_action_fn, get_sequence_fn


def make_eval_env():
    """Create a raw QWOP env with TimeLimit but no logging wrapper."""
    env = gym.make(
        "QWOP-v1",
        browser=CHROME_PATH,
        driver=CHROMEDRIVER_PATH,
        frames_per_step=FRAMES_PER_STEP,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    return env


def run_policy_eval(env, get_action_fn, num_episodes: int) -> list[dict]:
    """Evaluate a policy-based agent for num_episodes."""
    results = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_actions = []
        done = False

        while not done:
            action = get_action_fn(obs)
            episode_actions.append(int(action))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        distance = float(info.get("distance", 0.0))
        game_time = float(info.get("time", 0.0)) * GAME_TIME_MULTIPLIER
        is_success = bool(info.get("is_success", False))

        results.append({
            "episode": ep + 1,
            "distance": round(distance, 4),
            "game_time": round(game_time, 4),
            "is_success": is_success,
            "num_steps": len(episode_actions),
            "actions": episode_actions,
        })

        status = "FINISHED" if is_success else f"{distance:.1f}m"
        print(f"  Episode {ep + 1:3d}/{num_episodes}: {status} ({game_time:.1f}s game time)")

    return results


def run_sequence_eval(env, get_sequence_fn, num_episodes: int) -> list[dict]:
    """Evaluate an action-sequence agent for num_episodes.

    Since qwop-gym is deterministic, all episodes should produce identical results.
    We still run all of them to confirm determinism and for consistency with policy eval.
    """
    action_sequence = get_sequence_fn()
    results = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_actions = []
        done = False
        step_idx = 0

        while not done and step_idx < len(action_sequence):
            action = action_sequence[step_idx]
            episode_actions.append(int(action))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_idx += 1

        distance = float(info.get("distance", 0.0))
        game_time = float(info.get("time", 0.0)) * GAME_TIME_MULTIPLIER
        is_success = bool(info.get("is_success", False))

        results.append({
            "episode": ep + 1,
            "distance": round(distance, 4),
            "game_time": round(game_time, 4),
            "is_success": is_success,
            "num_steps": len(episode_actions),
            "actions": episode_actions,
        })

        status = "FINISHED" if is_success else f"{distance:.1f}m"
        print(f"  Episode {ep + 1:3d}/{num_episodes}: {status} ({game_time:.1f}s game time)")

    return results


def compute_summary(results: list[dict]) -> dict:
    """Compute aggregate stats from episode results."""
    distances = [r["distance"] for r in results]
    successes = [r for r in results if r["is_success"]]
    finish_times = [r["game_time"] for r in successes]

    return {
        "num_episodes": len(results),
        "mean_distance": round(float(np.mean(distances)), 2),
        "median_distance": round(float(np.median(distances)), 2),
        "max_distance": round(float(np.max(distances)), 2),
        "min_distance": round(float(np.min(distances)), 2),
        "std_distance": round(float(np.std(distances)), 2),
        "success_rate": round(len(successes) / len(results), 4),
        "races_finished": len(successes),
        "mean_finish_time": round(float(np.mean(finish_times)), 2) if finish_times else None,
        "best_finish_time": round(float(np.min(finish_times)), 2) if finish_times else None,
    }


def save_best_replay(results: list[dict], replays_dir: str, prefix: str) -> str | None:
    """Save the action replay for the best episode."""
    if not results:
        return None

    best = max(results, key=lambda r: r["distance"])
    filename = f"{prefix}_best_{best['distance']:.1f}m_ep{best['episode']}.json"
    filepath = os.path.join(replays_dir, filename)

    os.makedirs(replays_dir, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump({
            "episode": best["episode"],
            "distance": best["distance"],
            "game_time": best["game_time"],
            "is_success": best["is_success"],
            "actions": best["actions"],
        }, f, indent=2)

    return filepath


def main():
    parser = argparse.ArgumentParser(description="QWOP Autoresearch v2 — Evaluation Harness")
    parser.add_argument("--agent", required=True, help="Path to agent module (must have get_action or get_action_sequence)")
    parser.add_argument("--out", required=True, help="Path to save JSON results")
    parser.add_argument("--episodes", type=int, default=NUM_EVAL_EPISODES, help=f"Number of eval episodes (default: {NUM_EVAL_EPISODES})")
    parser.add_argument("--replays-dir", default="results/replays", help="Directory to save best replay")
    parser.add_argument("--label", default="agent", help="Label for this eval run (used in replay filename prefix)")
    args = parser.parse_args()

    print(f"Loading agent from: {args.agent}")
    get_action_fn, get_sequence_fn = load_agent(args.agent)

    agent_type = "policy" if get_action_fn else "sequence"
    print(f"Agent type: {agent_type}")
    print(f"Running {args.episodes} evaluation episodes...\n")

    env = make_eval_env()
    start_time = time.time()

    if get_action_fn:
        results = run_policy_eval(env, get_action_fn, args.episodes)
    else:
        results = run_sequence_eval(env, get_sequence_fn, args.episodes)

    wall_time = time.time() - start_time
    env.close()

    summary = compute_summary(results)
    summary["agent_path"] = args.agent
    summary["agent_type"] = agent_type
    summary["wall_clock_seconds"] = round(wall_time, 2)

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    output = {"summary": summary, "episodes": results}
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    # Save best replay
    replay_path = save_best_replay(results, args.replays_dir, args.label)

    # Print summary
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS — {args.label}")
    print(f"{'='*50}")
    print(f"  Episodes:         {summary['num_episodes']}")
    print(f"  Mean distance:    {summary['mean_distance']}m")
    print(f"  Median distance:  {summary['median_distance']}m")
    print(f"  Max distance:     {summary['max_distance']}m")
    print(f"  Std distance:     {summary['std_distance']}m")
    print(f"  Success rate:     {summary['success_rate']*100:.1f}%")
    print(f"  Races finished:   {summary['races_finished']}/{summary['num_episodes']}")
    if summary["mean_finish_time"]:
        print(f"  Mean finish time: {summary['mean_finish_time']}s")
        print(f"  Best finish time: {summary['best_finish_time']}s")
    print(f"  Wall clock:       {summary['wall_clock_seconds']:.0f}s")
    print(f"  Results saved:    {args.out}")
    if replay_path:
        print(f"  Best replay:      {replay_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
