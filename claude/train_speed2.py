"""
Speed training round 2 — even more aggressive time penalty.
Warm-starts from speed_model.pt (132.5s best).
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

SEED = 42
NUM_STEPS = 512
NUM_MINIBATCHES = 4
UPDATE_EPOCHS = 4
LEARNING_RATE = 3e-5  # very conservative fine-tuning
ANNEAL_LR = True
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.1  # tighter clipping to prevent catastrophic updates
ENT_COEF = 0.015  # slightly lower than default
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

STEP_BUDGET_THIS_PHASE = 1_500_000
CHECKPOINT_EVERY = 100

OBS_DIM = 60
ACT_DIM = 16


class SpeedRewardWrapper(gym.Wrapper):
    """Moderate speed bonus — not too aggressive to avoid destabilizing."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if (terminated or truncated) and info.get("is_success", False):
            game_time = info.get("time", 0.0) * 10
            # Linear bonus, scaled modestly
            speed_bonus = max(0, (180.0 - game_time)) * 0.5
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

    device = torch.device("cpu")

    raw_env = make_counted_env(time_cost_mult=30, success_reward=100)
    env = SpeedRewardWrapper(raw_env)

    agent = Agent().to(device)
    model_path = os.path.join(os.path.dirname(__file__), "speed_model.pt")
    if os.path.exists(model_path):
        agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Warm-starting from {model_path}")

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
    print(f"Speed training round 2: {num_iterations} iterations ({STEP_BUDGET_THIS_PHASE:,} steps)")
    print(f"Budget remaining: {get_remaining_steps():,}")
    print(f"Mods: time_cost=50, success_reward=150, quadratic speed bonus, ENT=0.01, LR=5e-5")

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

                if is_success:
                    print(f"  Ep {episode_count:4d}: FINISHED {game_time:.1f}s "
                          f"({len(current_episode_actions)} steps)")

                current_episode_actions = []
                next_obs, _ = env.reset()
                next_obs = torch.FloatTensor(next_obs).to(device)

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

        b_inds = np.arange(batch_size)
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    obs_buf[mb_inds], actions_buf.long()[mb_inds])
                logratio = newlogprob - logprobs_buf[mb_inds]
                ratio = logratio.exp()
                mb_adv = (advantages[mb_inds] - advantages[mb_inds].mean()) / (advantages[mb_inds].std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - returns[mb_inds]) ** 2).mean()
                loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        elapsed = time.time() - start_time
        sps = steps_this_phase / elapsed if elapsed > 0 else 0
        best_str = f"{best_time:.1f}s" if best_time < float("inf") else "N/A"
        print(f"Iter {iteration:4d}/{num_iterations} | {steps_this_phase:>8,} steps | "
              f"SPS: {sps:,.0f} | best_time: {best_str} | finishes: {finish_count}")

        if iteration % CHECKPOINT_EVERY == 0:
            _model_path = os.path.join(os.path.dirname(__file__), "speed2_model.pt")
            torch.save(agent.state_dict(), _model_path)
            if best_actions:
                _seq_path = os.path.join(os.path.dirname(__file__), "speed_best_sequence.json")
                with open(_seq_path, "w") as f:
                    json.dump({"game_time": float(best_time), "num_actions": len(best_actions),
                              "actions": best_actions}, f)
                print(f"  >> Checkpoint: best_time={best_time:.1f}s ({len(best_actions)} steps)")

    # Final save — to separate file to avoid clobbering working model
    _model_path = os.path.join(os.path.dirname(__file__), "speed2_model.pt")
    torch.save(agent.state_dict(), _model_path)
    if best_actions:
        _seq_path = os.path.join(os.path.dirname(__file__), "speed_best_sequence.json")
        with open(_seq_path, "w") as f:
            json.dump({"game_time": float(best_time), "num_actions": len(best_actions),
                      "actions": best_actions}, f)
    print(f"\nFinal: best_time={best_time:.1f}s, finishes={finish_count}/{episode_count}")
    env.close()
    print_budget_status()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train()
