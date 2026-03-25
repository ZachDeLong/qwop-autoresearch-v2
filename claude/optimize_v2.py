"""
Phase 1c (v2): Smarter sequence optimization.

Strategy:
1. Collect MANY stochastic rollouts from PPO policy to build a population
2. Keep only finishing sequences
3. For the best sequences, try targeted mutations:
   - Single-action replacement with policy-suggested alternatives
   - Crossover segments between good sequences
   - Tail truncation (many sequences may have unnecessary trailing steps)

The key insight from v1 hill climbing: random mutations in a 1200-step
gait sequence almost always cause the runner to fall. We need mutations
that are "in distribution" — actions the policy would actually consider.
"""

import sys
import os
import json
import time
import random
import copy

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step_counter import make_counted_env, get_step_count, get_remaining_steps, print_budget_status
from train_ppo import Agent

NUM_ACTIONS = 16


def collect_rollouts(env, agent, device, num_rollouts, deterministic=False):
    """Collect rollouts from the PPO policy."""
    results = []
    for i in range(num_rollouts):
        obs, _ = env.reset()
        obs_t = torch.FloatTensor(obs).to(device)
        actions = []
        done = False

        while not done:
            with torch.no_grad():
                if deterministic:
                    logits = agent.actor(obs_t.unsqueeze(0))
                    action = torch.argmax(logits, dim=-1)
                else:
                    action, _, _, _ = agent.get_action_and_value(obs_t.unsqueeze(0))

            a = int(action.item())
            actions.append(a)
            obs, reward, terminated, truncated, info = env.step(a)
            obs_t = torch.FloatTensor(obs).to(device)
            done = terminated or truncated

        dist = float(info.get("distance", 0.0))
        success = bool(info.get("is_success", False))
        gt = float(info.get("time", 0.0)) * 10

        results.append({
            "distance": dist,
            "is_success": success,
            "game_time": gt,
            "num_steps": len(actions),
            "actions": actions,
        })

        status = "FINISHED" if success else f"{dist:.1f}m"
        if (i + 1) % 50 == 0 or success:
            print(f"  Rollout {i+1}/{num_rollouts}: {status} ({gt:.1f}s, {len(actions)} steps)")

    return results


def evaluate_sequence(env, actions):
    """Play an action sequence. Returns (distance, is_success, game_time, actual_steps)."""
    obs, _ = env.reset()
    done = False
    step_idx = 0

    while not done and step_idx < len(actions):
        obs, reward, terminated, truncated, info = env.step(actions[step_idx])
        done = terminated or truncated
        step_idx += 1

    distance = float(info.get("distance", 0.0))
    is_success = bool(info.get("is_success", False))
    game_time = float(info.get("time", 0.0)) * 10

    return distance, is_success, game_time, step_idx


def crossover(seq_a, seq_b):
    """Single-point crossover between two sequences."""
    min_len = min(len(seq_a), len(seq_b))
    point = random.randint(min_len // 4, 3 * min_len // 4)
    child1 = seq_a[:point] + seq_b[point:]
    child2 = seq_b[:point] + seq_a[point:]
    return child1, child2


def policy_guided_mutation(env, agent, device, sequence, num_positions=5):
    """Mutate a sequence using policy-suggested alternative actions.

    For each mutated position:
    1. Replay the sequence up to that position to get the observation
    2. Query the PPO policy for its action distribution
    3. Sample from the policy (instead of random)
    """
    mutated = sequence.copy()
    positions = random.sample(range(len(sequence)), min(num_positions, len(sequence)))
    positions.sort()

    # Replay to each position and get policy suggestion
    obs, _ = env.reset()
    obs_t = torch.FloatTensor(obs).to(device)
    current_pos = 0

    for target_pos in positions:
        # Advance to target position
        while current_pos < target_pos:
            obs, reward, terminated, truncated, info = env.step(mutated[current_pos])
            obs_t = torch.FloatTensor(obs).to(device)
            if terminated or truncated:
                # Episode ended before we got to the mutation point
                return mutated[:current_pos + 1]
            current_pos += 1

        # At target position, sample from policy
        with torch.no_grad():
            logits = agent.actor(obs_t.unsqueeze(0)).squeeze(0)
            probs = torch.softmax(logits, dim=-1)
            # Sample a different action from the policy distribution
            action = torch.multinomial(probs, 1).item()
            mutated[target_pos] = action

    return mutated


def save_checkpoint(seq, distance, success, game_time, label="optimized"):
    """Save current best sequence."""
    for filename in ["best_sequence.json", "optimized_sequence.json"]:
        path = os.path.join(os.path.dirname(__file__), filename)
        with open(path, "w") as f:
            json.dump({
                "distance": float(distance),
                "is_success": bool(success),
                "game_time": float(game_time),
                "num_actions": len(seq),
                "source": label,
                "actions": [int(a) for a in seq],
            }, f, indent=2)


def optimize():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cpu")

    # Load PPO model
    model_path = os.path.join(os.path.dirname(__file__), "ppo_model.pt")
    agent = Agent().to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    agent.eval()
    print(f"Loaded PPO model")

    env = make_counted_env()

    # ======================================================
    # Phase A: Collect a large population of stochastic rollouts
    # Budget: ~2M steps (estimate: 2000 rollouts × ~1000 avg steps)
    # ======================================================
    print(f"\n{'='*50}")
    print("PHASE A: Mass rollout collection")
    print(f"{'='*50}")
    print(f"Budget remaining: {get_remaining_steps():,}")

    all_results = collect_rollouts(env, agent, device, num_rollouts=2000)

    finishing = [r for r in all_results if r["is_success"]]
    print(f"\nCollected {len(all_results)} rollouts:")
    print(f"  Finishing: {len(finishing)} ({100*len(finishing)/len(all_results):.0f}%)")

    if finishing:
        distances = [r["distance"] for r in finishing]
        times = [r["game_time"] for r in finishing]
        steps = [r["num_steps"] for r in finishing]
        print(f"  Best distance: {max(distances):.1f}m")
        print(f"  Best time: {min(times):.1f}s ({min(steps)} steps)")
        print(f"  Mean distance: {np.mean(distances):.1f}m")
        print(f"  Mean time: {np.mean(times):.1f}s")

        # Sort by distance (descending), then by steps (ascending)
        finishing.sort(key=lambda r: (-r["distance"], r["num_steps"]))
        best = finishing[0]
        print(f"\n  BEST: {best['distance']:.1f}m in {best['num_steps']} steps ({best['game_time']:.1f}s)")

        save_checkpoint(best["actions"], best["distance"], best["is_success"],
                       best["game_time"], label="mass_rollout_best")
    else:
        print("  No finishing rollouts! Using deterministic rollout.")
        # Fall back to deterministic
        det_results = collect_rollouts(env, agent, device, num_rollouts=1, deterministic=True)
        best = det_results[0]
        save_checkpoint(best["actions"], best["distance"], best["is_success"],
                       best.get("game_time", 0), label="deterministic_fallback")

    print_budget_status()

    # ======================================================
    # Phase B: Crossover between top finishing sequences
    # Budget: ~2M steps
    # ======================================================
    remaining = get_remaining_steps()
    if remaining > 500_000 and len(finishing) >= 2:
        print(f"\n{'='*50}")
        print("PHASE B: Crossover optimization")
        print(f"{'='*50}")

        top_seqs = finishing[:min(20, len(finishing))]  # top 20 sequences
        best_dist = best["distance"]
        best_seq = best["actions"]
        best_success = best["is_success"]
        best_gt = best.get("game_time", 0)

        num_crossover_evals = 0
        crossover_improvements = 0

        while get_remaining_steps() > 3_000_000:  # save budget for Phase C
            # Pick two parents from top sequences
            p1 = random.choice(top_seqs)
            p2 = random.choice(top_seqs)
            while p2 is p1 and len(top_seqs) > 1:
                p2 = random.choice(top_seqs)

            child1, child2 = crossover(p1["actions"], p2["actions"])

            for child in [child1, child2]:
                dist, success, gt, steps = evaluate_sequence(env, child)
                num_crossover_evals += 1

                if dist > best_dist or (dist == best_dist and success and not best_success):
                    crossover_improvements += 1
                    best_dist = dist
                    best_seq = child
                    best_success = success
                    best_gt = gt
                    save_checkpoint(best_seq, best_dist, best_success, best_gt,
                                  label=f"crossover_eval{num_crossover_evals}")
                    print(f"  *** Crossover improvement #{crossover_improvements}: "
                          f"{dist:.1f}m ({steps} steps, {gt:.1f}s)")

                    # Add successful child to the pool
                    if success:
                        top_seqs.append({
                            "distance": dist, "is_success": success,
                            "game_time": gt, "num_steps": len(child),
                            "actions": child,
                        })

                if num_crossover_evals % 200 == 0:
                    print(f"  Crossover eval {num_crossover_evals}: "
                          f"best={best_dist:.1f}m, improvements={crossover_improvements}, "
                          f"budget={get_remaining_steps():,}")

        print(f"\nCrossover done: {num_crossover_evals} evals, {crossover_improvements} improvements")
        print(f"  Best: {best_dist:.1f}m")
        print_budget_status()

    # ======================================================
    # Phase C: Policy-guided refinement of best sequence
    # Budget: remaining ~3M steps
    # ======================================================
    remaining = get_remaining_steps()
    if remaining > 100_000:
        print(f"\n{'='*50}")
        print("PHASE C: Policy-guided refinement")
        print(f"{'='*50}")

        # Load current best
        with open(os.path.join(os.path.dirname(__file__), "best_sequence.json")) as f:
            data = json.load(f)
        current_seq = data["actions"]
        current_dist = data["distance"]
        current_success = data.get("is_success", False)
        current_gt = data.get("game_time", 0)

        print(f"Starting from: {current_dist:.1f}m ({len(current_seq)} steps)")

        improvements = 0
        iteration = 0

        while get_remaining_steps() > 10_000:
            iteration += 1

            # Try policy-guided mutation with varying intensity
            num_positions = random.choice([1, 1, 1, 2, 2, 3, 5])
            mutated = policy_guided_mutation(env, agent, device, current_seq,
                                           num_positions=num_positions)
            dist, success, gt, steps = evaluate_sequence(env, mutated)

            improved = False
            if dist > current_dist:
                improved = True
            elif dist == current_dist and success and not current_success:
                improved = True
            elif (dist >= current_dist - 0.5 and success and current_success
                  and gt < current_gt - 1.0):
                # Accept slightly shorter distance if significantly faster
                improved = True

            if improved:
                improvements += 1
                current_seq = mutated
                current_dist = dist
                current_success = success
                current_gt = gt
                save_checkpoint(current_seq, current_dist, current_success,
                              current_gt, label=f"refined_iter{iteration}")
                print(f"  *** Refinement #{improvements}: {dist:.1f}m "
                      f"({len(mutated)} steps, {gt:.1f}s)")

            if iteration % 100 == 0:
                print(f"  Refinement iter {iteration}: "
                      f"current={current_dist:.1f}m, improvements={improvements}, "
                      f"budget={get_remaining_steps():,}")

        print(f"\nRefinement done: {iteration} iterations, {improvements} improvements")
        print(f"  Final: {current_dist:.1f}m")

    env.close()
    print(f"\n{'='*50}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*50}")
    print_budget_status()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    optimize()
