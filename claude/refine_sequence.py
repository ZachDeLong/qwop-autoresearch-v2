"""
Targeted single-action refinement of the best sequence.

For a subset of positions in the sequence:
1. Replay the sequence up to that position
2. Try replacing that one action with each of the other 15 options
3. Play out the remainder of the original sequence
4. If any replacement leads to higher distance, keep it

This is a systematic search at each position, leveraging determinism.
The key insight: we only need to find ONE better action at ONE position
to improve the overall sequence.

Since replaying from scratch is expensive, we use a chunked approach:
- Divide the sequence into chunks
- For each chunk start, replay to that point (shared prefix)
- Then test alternatives for actions within that chunk
"""

import sys
import os
import json
import time
import random

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step_counter import make_counted_env, get_step_count, get_remaining_steps, print_budget_status

NUM_ACTIONS = 16
# How much budget to reserve (don't use ALL remaining)
BUDGET_RESERVE = 100_000


def evaluate_sequence(env, actions):
    """Play an action sequence. Returns (distance, is_success, game_time, actual_steps)."""
    obs, _ = env.reset()
    done = False
    step_idx = 0
    info = {}

    while not done and step_idx < len(actions):
        obs, reward, terminated, truncated, info = env.step(actions[step_idx])
        done = terminated or truncated
        step_idx += 1

    distance = float(info.get("distance", 0.0))
    is_success = bool(info.get("is_success", False))
    game_time = float(info.get("time", 0.0)) * 10

    return distance, is_success, game_time, step_idx


def replay_prefix(env, actions, up_to):
    """Replay the first `up_to` actions and return the observation.
    Returns (obs, done, info). If done=True, the episode ended before up_to."""
    obs, _ = env.reset()
    done = False
    info = {}

    for i in range(up_to):
        obs, reward, terminated, truncated, info = env.step(actions[i])
        done = terminated or truncated
        if done:
            break

    return obs, done, info


def evaluate_with_replacement(env, original_seq, position, new_action):
    """Evaluate the sequence with one action replaced at position.
    Replays from start, applies replacement, then continues with original."""
    obs, _ = env.reset()
    done = False
    info = {}

    for i in range(len(original_seq)):
        if done:
            break
        if i == position:
            obs, reward, terminated, truncated, info = env.step(new_action)
        else:
            obs, reward, terminated, truncated, info = env.step(original_seq[i])
        done = terminated or truncated

    distance = float(info.get("distance", 0.0))
    is_success = bool(info.get("is_success", False))
    game_time = float(info.get("time", 0.0)) * 10
    return distance, is_success, game_time


def save_checkpoint(seq, distance, success, game_time, label):
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


def refine():
    random.seed(123)
    np.random.seed(123)

    # Load best sequence
    seq_path = os.path.join(os.path.dirname(__file__), "best_sequence.json")
    with open(seq_path) as f:
        data = json.load(f)

    current_seq = data["actions"]
    current_dist = data["distance"]
    current_success = data.get("is_success", False)
    current_gt = data.get("game_time", 0)
    print(f"Starting sequence: {current_dist:.1f}m ({len(current_seq)} steps)")

    env = make_counted_env()

    # Verify
    print("Verifying starting sequence...")
    dist, success, gt, steps = evaluate_sequence(env, current_seq)
    print(f"  Verified: {dist:.1f}m (success={success}, {steps} steps, {gt:.1f}s)")
    current_dist = dist
    current_success = success
    current_gt = gt

    start_time = time.time()
    improvements = 0
    positions_tested = 0
    total_evals = 0

    # Strategy: test random positions with all 15 alternative actions
    # Each test costs len(current_seq) steps per alternative
    # Cost per position: 15 * len(current_seq) ≈ 15 * 1291 = 19,365 steps
    # With ~7M budget: ~361 positions testable

    # Create shuffled position list (test in random order to spread exploration)
    positions = list(range(len(current_seq)))
    random.shuffle(positions)

    print(f"\nStarting single-action refinement...")
    print(f"  Sequence length: {len(current_seq)}")
    print(f"  Budget remaining: {get_remaining_steps():,}")
    print(f"  Estimated positions testable: {get_remaining_steps() // (15 * len(current_seq))}")

    for pos in positions:
        if get_remaining_steps() <= BUDGET_RESERVE:
            print(f"\nBudget reserve reached. Stopping.")
            break

        original_action = current_seq[pos]
        best_replacement = None
        best_replacement_dist = current_dist

        # Try all alternative actions at this position
        for alt_action in range(NUM_ACTIONS):
            if alt_action == original_action:
                continue

            if get_remaining_steps() <= BUDGET_RESERVE:
                break

            dist, success, gt = evaluate_with_replacement(
                env, current_seq, pos, alt_action
            )
            total_evals += 1

            if dist > best_replacement_dist:
                best_replacement_dist = dist
                best_replacement = alt_action
                best_success = success
                best_gt = gt

        positions_tested += 1

        if best_replacement is not None:
            improvements += 1
            old_action = current_seq[pos]
            current_seq[pos] = best_replacement
            current_dist = best_replacement_dist
            current_success = best_success
            current_gt = best_gt

            save_checkpoint(current_seq, current_dist, current_success,
                          current_gt, f"refined_pos{pos}_a{old_action}to{best_replacement}")

            print(f"  *** Position {pos}: action {old_action}->{best_replacement}, "
                  f"dist {current_dist:.1f}m (was {best_replacement_dist:.1f}m before)")

        if positions_tested % 25 == 0:
            elapsed = time.time() - start_time
            print(f"  Tested {positions_tested} positions | "
                  f"{total_evals} evals | "
                  f"{improvements} improvements | "
                  f"best={current_dist:.1f}m | "
                  f"budget={get_remaining_steps():,} | "
                  f"{elapsed:.0f}s")

    env.close()

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"REFINEMENT COMPLETE")
    print(f"{'='*50}")
    print(f"  Positions tested: {positions_tested}")
    print(f"  Total evaluations: {total_evals}")
    print(f"  Improvements: {improvements}")
    print(f"  Final distance: {current_dist:.1f}m")
    print(f"  Final success: {current_success}")
    print(f"  Wall time: {elapsed:.0f}s")
    print(f"{'='*50}")

    print_budget_status()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    refine()
