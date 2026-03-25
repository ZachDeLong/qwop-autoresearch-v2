"""
Phase 1b: Extract best action sequences from trained PPO policy.

Runs the policy both deterministically (argmax) and stochastically,
collecting the best action sequence for optimization.
"""

import sys
import os
import json
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step_counter import make_counted_env, get_step_count, get_remaining_steps, print_budget_status
from train_ppo import Agent

NUM_STOCHASTIC_ROLLOUTS = 5
NUM_DETERMINISTIC_ROLLOUTS = 2


def run_rollout(env, agent, device, deterministic=False):
    """Run one episode, return (distance, is_success, game_time, actions)."""
    obs, _ = env.reset()
    obs = torch.FloatTensor(obs).to(device)
    actions = []
    done = False

    while not done:
        with torch.no_grad():
            if deterministic:
                action = agent.get_deterministic_action(obs.unsqueeze(0))
            else:
                action, _, _, _ = agent.get_action_and_value(obs.unsqueeze(0))

        action_int = int(action.item())
        actions.append(action_int)
        obs, reward, terminated, truncated, info = env.step(action_int)
        obs = torch.FloatTensor(obs).to(device)
        done = terminated or truncated

    distance = info.get("distance", 0.0)
    is_success = info.get("is_success", False)
    game_time = info.get("time", 0.0) * 10

    return distance, is_success, game_time, actions


def extract():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_path = os.path.join(os.path.dirname(__file__), "ppo_model.pt")
    agent = Agent().to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.eval()
    print(f"Loaded model from {model_path}")

    env = make_counted_env()

    all_results = []

    # Deterministic rollouts
    print(f"\n--- Deterministic rollouts ({NUM_DETERMINISTIC_ROLLOUTS}) ---")
    for i in range(NUM_DETERMINISTIC_ROLLOUTS):
        dist, success, gt, actions = run_rollout(env, agent, device, deterministic=True)
        status = "FINISHED" if success else f"{dist:.1f}m"
        print(f"  Det {i+1}: {status} ({gt:.1f}s, {len(actions)} steps)")
        all_results.append({
            "type": "deterministic",
            "distance": float(dist),
            "is_success": success,
            "game_time": float(gt),
            "actions": actions,
        })

    # Stochastic rollouts
    print(f"\n--- Stochastic rollouts ({NUM_STOCHASTIC_ROLLOUTS}) ---")
    for i in range(NUM_STOCHASTIC_ROLLOUTS):
        dist, success, gt, actions = run_rollout(env, agent, device, deterministic=False)
        status = "FINISHED" if success else f"{dist:.1f}m"
        print(f"  Sto {i+1}: {status} ({gt:.1f}s, {len(actions)} steps)")
        all_results.append({
            "type": "stochastic",
            "distance": float(dist),
            "is_success": success,
            "game_time": float(gt),
            "actions": actions,
        })

    env.close()

    # Also load any sequence from PPO training
    ppo_seq_path = os.path.join(os.path.dirname(__file__), "ppo_best_sequence.json")
    if os.path.exists(ppo_seq_path):
        with open(ppo_seq_path) as f:
            ppo_best = json.load(f)
        all_results.append({
            "type": "ppo_training_best",
            "distance": ppo_best["distance"],
            "is_success": ppo_best["distance"] >= 100.0,
            "game_time": 0,
            "actions": ppo_best["actions"],
        })
        print(f"\nPPO training best: {ppo_best['distance']:.1f}m ({ppo_best['num_actions']} steps)")

    # Pick the best sequence
    best = max(all_results, key=lambda r: r["distance"])
    print(f"\n=== BEST SEQUENCE ===")
    print(f"  Type: {best['type']}")
    print(f"  Distance: {best['distance']:.1f}m")
    print(f"  Success: {best['is_success']}")
    print(f"  Steps: {len(best['actions'])}")

    # Save it
    out_path = os.path.join(os.path.dirname(__file__), "best_sequence.json")
    with open(out_path, "w") as f:
        json.dump({
            "distance": float(best["distance"]),
            "is_success": bool(best["is_success"]),
            "num_actions": len(best["actions"]),
            "source": best["type"],
            "actions": best["actions"],
        }, f, indent=2)
    print(f"  Saved: {out_path}")

    # Save all results summary
    summary_path = os.path.join(os.path.dirname(__file__), "extraction_summary.json")
    summary = [{
        "type": r["type"],
        "distance": float(r["distance"]),
        "is_success": bool(r["is_success"]),
        "num_steps": len(r["actions"]),
    } for r in all_results]
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_budget_status()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    extract()
