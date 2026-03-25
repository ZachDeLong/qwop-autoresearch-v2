"""
Speed-optimized PPO training for QWOP.

Goal: minimize finish time rather than maximize distance.
Strategy:
1. Use higher time_cost_mult to penalize slow running
2. Use higher success_reward to strongly incentivize finishing
3. Add a custom reward wrapper that gives bonus for fast finishes
4. Start from the existing PPO model (warm start)
"""

import sys
import os
import json
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step_counter import make_counted_env, get_step_count, get_remaining_steps, print_budget_status

# === HYPERPARAMETERS ===
SEED = 1
NUM_STEPS = 512
NUM_MINIBATCHES = 4
UPDATE_EPOCHS = 4
LEARNING_RATE = 1e-4  # lower LR since we're fine-tuning
ANNEAL_LR = True
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.02
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = None

STEP_BUDGET_THIS_PHASE = 1_500_000
CHECKPOINT_EVERY = 100

OBS_DIM = 60
ACT_DIM = 16


class SpeedRewardWrapper(gym.Wrapper):
    """Wrapper that modifies rewards to incentivize speed.
    Adds a bonus for finishing that's inversely proportional to game time."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # If the episode ended successfully, add a speed bonus
        if (terminated or truncated) and info.get("is_success", False):
            game_time = info.get("time", 0.0) * 10  # convert to seconds
            # Bonus: higher for faster finishes. Baseline ~160s, target <120s
            # speed_bonus = max(0, (200 - game_time) / 10)
            speed_bonus = 200.0 / max(game_time, 1.0)  # inversely proportional
            reward += speed_bonus

        return obs, reward, terminated, truncated, info


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(OBS_DIM, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(OBS_DIM, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, ACT_DIM),
        )
        for layer in [*self.critic, *self.actor]:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create env with higher time cost and success reward
    raw_env = make_counted_env(time_cost_mult=30, success_reward=100)
    env = SpeedRewardWrapper(raw_env)

    # Load existing model as warm start
    agent = Agent().to(device)
    model_path = os.path.join(os.path.dirname(__file__), "ppo_model.pt")
    if os.path.exists(model_path):
        agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Warm-starting from {model_path}")
    else:
        print("No existing model found, training from scratch")

    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    batch_size = NUM_STEPS
    minibatch_size = batch_size // NUM_MINIBATCHES

    obs_buf = torch.zeros((NUM_STEPS,) + (OBS_DIM,)).to(device)
    actions_buf = torch.zeros(NUM_STEPS).to(device)
    logprobs_buf = torch.zeros(NUM_STEPS).to(device)
    rewards_buf = torch.zeros(NUM_STEPS).to(device)
    dones_buf = torch.zeros(NUM_STEPS).to(device)
    values_buf = torch.zeros(NUM_STEPS).to(device)

    start_time = time.time()
    steps_this_phase = 0
    episode_count = 0
    best_time = float("inf")
    best_actions = []
    current_episode_actions = []
    finish_count = 0

    next_obs, _ = env.reset()
    next_obs = torch.FloatTensor(next_obs).to(device)
    next_done = torch.tensor(0.0).to(device)

    num_iterations = STEP_BUDGET_THIS_PHASE // batch_size
    print(f"Training for {num_iterations} iterations ({STEP_BUDGET_THIS_PHASE:,} steps)")
    print(f"Budget remaining: {get_remaining_steps():,}")
    print(f"Reward mods: time_cost_mult=30, success_reward=100, speed_bonus=200/time")

    for iteration in range(1, num_iterations + 1):
        if ANNEAL_LR:
            frac = 1.0 - (iteration - 1) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * LEARNING_RATE

        for step in range(NUM_STEPS):
            steps_this_phase += 1
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
                values_buf[step] = value.flatten()
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            current_episode_actions.append(int(action.item()))

            next_obs, reward, terminated, truncated, info = env.step(int(action.item()))
            done = terminated or truncated
            rewards_buf[step] = torch.tensor(reward).to(device)
            next_obs = torch.FloatTensor(next_obs).to(device)
            next_done = torch.tensor(float(done)).to(device)

            if done:
                episode_count += 1
                distance = float(info.get("distance", 0.0))
                is_success = bool(info.get("is_success", False))
                game_time = float(info.get("time", 0.0)) * 10

                if is_success:
                    finish_count += 1
                    if game_time < best_time:
                        best_time = game_time
                        best_actions = current_episode_actions.copy()

                status = f"FINISHED {game_time:.1f}s" if is_success else f"{distance:.1f}m"
                if is_success or episode_count % 50 == 0:
                    print(f"  Ep {episode_count:4d}: {status} "
                          f"({len(current_episode_actions)} steps, "
                          f"finishes={finish_count})")

                current_episode_actions = []
                next_obs, _ = env.reset()
                next_obs = torch.FloatTensor(next_obs).to(device)

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).reshape(1)
            advantages = torch.zeros_like(rewards_buf).to(device)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + GAMMA * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        b_obs = obs_buf
        b_logprobs = logprobs_buf
        b_actions = actions_buf
        b_advantages = advantages
        b_returns = returns

        b_inds = np.arange(batch_size)
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        elapsed = time.time() - start_time
        sps = steps_this_phase / elapsed if elapsed > 0 else 0
        best_str = f"{best_time:.1f}s" if best_time < float("inf") else "N/A"
        print(f"Iter {iteration:4d}/{num_iterations} | {steps_this_phase:>8,} steps | "
              f"{elapsed:>6.0f}s | SPS: {sps:,.0f} | loss: {loss.item():.4f} | "
              f"best_time: {best_str} | finishes: {finish_count}")

        if iteration % CHECKPOINT_EVERY == 0:
            _save_checkpoint(agent, best_actions, best_time, iteration, "speed")

    _save_checkpoint(agent, best_actions, best_time, iteration, "speed_final")
    env.close()
    print_budget_status()


def _save_checkpoint(agent, best_actions, best_time, iteration, label):
    model_path = os.path.join(os.path.dirname(__file__), "speed_model.pt")
    torch.save(agent.state_dict(), model_path)

    if best_actions and best_time < float("inf"):
        seq_path = os.path.join(os.path.dirname(__file__), "speed_best_sequence.json")
        with open(seq_path, "w") as f:
            json.dump({
                "game_time": float(best_time),
                "num_actions": len(best_actions),
                "actions": best_actions,
            }, f)
        print(f"  >> Checkpoint: model saved, best_time={best_time:.1f}s ({len(best_actions)} steps)")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train()
