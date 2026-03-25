"""
Phase 1a: PPO training for QWOP.

Uses v1's proven best config (128x128 Tanh, NUM_STEPS=512, ENT_COEF=0.02).
Trains for ~2M steps to get a solid policy for sequence extraction.
"""

import sys
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# Add parent dir so we can import step_counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step_counter import make_counted_env, get_step_count, get_remaining_steps, print_budget_status

# === HYPERPARAMETERS (v1 best config) ===
SEED = 1
NUM_STEPS = 512
NUM_MINIBATCHES = 4
UPDATE_EPOCHS = 4
LEARNING_RATE = 2.5e-4
ANNEAL_LR = True
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.02
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = None

# Training budget for this phase
STEP_BUDGET_THIS_PHASE = 500_000
CHECKPOINT_EVERY = 100  # save model + best sequence every N iterations

# === NETWORK (v1 best: 128x128 Tanh) ===
OBS_DIM = 60
ACT_DIM = 16


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
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

    def get_deterministic_action(self, x):
        """Return argmax action (no sampling)."""
        logits = self.actor(x)
        return torch.argmax(logits, dim=-1)


def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create env with step counting
    env = make_counted_env()
    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # Rollout storage
    batch_size = NUM_STEPS
    minibatch_size = batch_size // NUM_MINIBATCHES

    obs_buf = torch.zeros((NUM_STEPS,) + (OBS_DIM,)).to(device)
    actions_buf = torch.zeros(NUM_STEPS).to(device)
    logprobs_buf = torch.zeros(NUM_STEPS).to(device)
    rewards_buf = torch.zeros(NUM_STEPS).to(device)
    dones_buf = torch.zeros(NUM_STEPS).to(device)
    values_buf = torch.zeros(NUM_STEPS).to(device)

    # Tracking
    start_time = time.time()
    initial_steps = get_step_count()
    steps_this_phase = 0
    episode_count = 0
    best_distance = 0.0
    best_actions = []
    current_episode_actions = []

    next_obs, _ = env.reset()
    next_obs = torch.FloatTensor(next_obs).to(device)
    next_done = torch.tensor(0.0).to(device)

    num_iterations = STEP_BUDGET_THIS_PHASE // batch_size
    print(f"Training for {num_iterations} iterations ({STEP_BUDGET_THIS_PHASE:,} steps)")
    print(f"Budget remaining: {get_remaining_steps():,}")

    for iteration in range(1, num_iterations + 1):
        # LR annealing
        if ANNEAL_LR:
            frac = 1.0 - (iteration - 1) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * LEARNING_RATE

        # === ROLLOUT ===
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
                distance = info.get("distance", 0.0)
                is_success = info.get("is_success", False)
                game_time = info.get("time", 0.0) * 10

                if distance > best_distance:
                    best_distance = distance
                    best_actions = current_episode_actions.copy()

                status = "FINISHED!" if is_success else f"{distance:.1f}m"
                print(f"  Ep {episode_count:4d}: {status} ({game_time:.1f}s, {len(current_episode_actions)} steps)")

                current_episode_actions = []
                next_obs, _ = env.reset()
                next_obs = torch.FloatTensor(next_obs).to(device)

        # === GAE ===
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

        # Flatten
        b_obs = obs_buf
        b_logprobs = logprobs_buf
        b_actions = actions_buf
        b_advantages = advantages
        b_returns = returns
        b_values = values_buf

        # === PPO UPDATE ===
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

            if TARGET_KL is not None:
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    if approx_kl > TARGET_KL:
                        break

        # Progress
        elapsed = time.time() - start_time
        total_steps = get_step_count()
        sps = steps_this_phase / elapsed if elapsed > 0 else 0
        print(f"Iter {iteration:4d}/{num_iterations} | {steps_this_phase:>8,} steps | "
              f"{elapsed:>6.0f}s | SPS: {sps:,.0f} | loss: {loss.item():.4f} | "
              f"best: {best_distance:.1f}m")

        # Periodic checkpoint save
        if iteration % CHECKPOINT_EVERY == 0:
            import json
            _model_path = os.path.join(os.path.dirname(__file__), "ppo_model.pt")
            torch.save(agent.state_dict(), _model_path)
            if best_actions:
                _seq_path = os.path.join(os.path.dirname(__file__), "ppo_best_sequence.json")
                with open(_seq_path, "w") as f:
                    json.dump({
                        "distance": float(best_distance),
                        "num_actions": len(best_actions),
                        "actions": best_actions,
                    }, f)
                print(f"  >> Checkpoint: model + sequence saved (best: {best_distance:.1f}m)")

    # Save model
    model_path = os.path.join(os.path.dirname(__file__), "ppo_model.pt")
    torch.save(agent.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")

    # Save best action sequence from training
    if best_actions:
        import json
        seq_path = os.path.join(os.path.dirname(__file__), "ppo_best_sequence.json")
        with open(seq_path, "w") as f:
            json.dump({
                "distance": float(best_distance),
                "num_actions": len(best_actions),
                "actions": best_actions,
            }, f, indent=2)
        print(f"Best sequence saved: {seq_path} ({best_distance:.1f}m, {len(best_actions)} actions)")

    env.close()
    print_budget_status()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train()
