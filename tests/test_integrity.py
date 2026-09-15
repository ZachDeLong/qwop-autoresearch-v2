import multiprocessing

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from torch import nn

from qwop_lab.agents import LegacyPolicy, import_legacy_weights, load_policy
from qwop_lab.artifacts import digest
from qwop_lab.budget import BudgetExhausted, Ledger, StepMeter
from qwop_lab.environment import Case, reset_case
from qwop_lab.evaluation import rollout, select_replay, summarize
from qwop_lab.replay import frame_repeats, verify_sample


class ToyEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (60,), np.float32)
    action_space = gym.spaces.Discrete(16)

    def __init__(self):
        self.steps = 0
        self.resets = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.resets.append(seed)
        self.steps = 0
        return np.zeros(60, np.float32), {"time": 0.0, "distance": 0.0, "is_success": False}

    def step(self, action):
        self.steps += 1
        done = self.steps == 3
        return (
            np.full(60, self.steps / 10, np.float32),
            1.0,
            done,
            False,
            {
                "time": self.steps / 75,
                "distance": self.steps * 34,
                "is_success": done,
            },
        )


def _spend(path, count):
    lease = Ledger(path).lease("worker", block=8)
    for _ in range(count):
        lease.consume()
    lease.close()


def test_budget_is_shared_across_spawned_processes(tmp_path):
    path = str(tmp_path / "budget.sqlite")
    ledger = Ledger(path, 200)
    ctx = multiprocessing.get_context("spawn")
    processes = [ctx.Process(target=_spend, args=(path, 100)) for _ in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    assert ledger.status()["charged_steps"] == 200
    assert ledger.status()["confirmed_steps"] == 200


def test_budget_stops_before_sending_action(tmp_path):
    ledger = Ledger(tmp_path / "budget.sqlite", 2)
    raw = ToyEnv()
    env = StepMeter(raw, ledger.lease("trial", block=256))
    env.reset()
    env.step(0)
    env.step(0)
    with pytest.raises(BudgetExhausted):
        env.step(0)
    assert raw.steps == 2
    env.close()
    assert ledger.status()["remaining_steps"] == 0


def test_abandoned_lease_stays_conservatively_charged(tmp_path):
    ledger = Ledger(tmp_path / "budget.sqlite", 10)
    abandoned = ledger.lease("crashed", block=8)
    abandoned.consume()
    survivor = ledger.lease("next", block=8)
    survivor.consume()
    survivor.consume()
    with pytest.raises(BudgetExhausted):
        survivor.consume()
    survivor.close()
    assert ledger.status()["charged_steps"] == 10
    assert ledger.status()["confirmed_steps"] == 2


def test_clean_close_returns_unused_reservation(tmp_path):
    ledger = Ledger(tmp_path / "budget.sqlite", 20)
    lease = ledger.lease("trial", block=16)
    lease.consume()
    assert ledger.status()["charged_steps"] == 16
    lease.close()
    lease.close()
    assert ledger.status()["charged_steps"] == 1
    with pytest.raises(RuntimeError):
        lease.consume()
    with pytest.raises(ValueError):
        Ledger(ledger.path, 21)


def test_reset_case_recreates_explicit_reset_history():
    env = ToyEnv()
    reset_case(env, Case(42, 2))
    assert env.resets == [42, None, None]
    for args in ((0, 0), (42, -1), (42, 100)):
        with pytest.raises(ValueError):
            Case(*args)


def test_rollout_and_summary_report_duplicates_honestly():
    episodes = [rollout(ToyEnv(), lambda obs: 0, Case(seed)) for seed in (1, 2)]
    summary = summarize(episodes)
    assert summary["finishes"] == 2
    assert summary["unique_action_sequences"] == 1
    assert summary["unique_trajectories"] == 1
    assert summary["total_steps"] == 6
    assert episodes[0]["actions_sha256"] == digest([0, 0, 0])
    assert episodes[0]["first_100m_score_time"] == pytest.approx(0.4)


def test_select_replay_prioritizes_finish_then_speed():
    slow = {"is_success": True, "score_time": 160, "distance": 105}
    fast = {"is_success": True, "score_time": 130, "distance": 100.2}
    fall = {"is_success": False, "score_time": 5, "distance": 20}
    assert select_replay([slow, fast, fall]) is fast
    assert select_replay([fall]) is fall
    with pytest.raises(ValueError):
        summarize([])


def test_replay_rejects_state_or_termination_divergence():
    episode = rollout(ToyEnv(), lambda obs: 0, Case(1))
    original = episode["trace"][-1]
    verify_sample(original, original, 3)
    for key, wrong in (
        ("obs_sha256", "bad"),
        ("raw_time", 17),
        ("terminated", False),
        ("distance", 103),
        ("is_success", False),
    ):
        with pytest.raises(ValueError, match=key):
            verify_sample({**original, key: wrong}, original, 3)


def test_video_timestamps_follow_simulation_clock_without_accumulated_drift():
    times = [0.016666 + n * 4 / 30 for n in range(1001)]
    assert sum(frame_repeats(a, b) for a, b in zip(times, times[1:])) == 4000
    with pytest.raises(ValueError):
        frame_repeats(10, 10)


def test_reset_clock_offset_does_not_allow_subsequent_clock_drift():
    episode = rollout(ToyEnv(), lambda obs: 0, Case(1))
    expected = episode["trace"][0]
    offset = 0.00003
    shifted = {**expected, "raw_time": expected["raw_time"] + offset}
    verify_sample(shifted, expected, 1, offset)
    with pytest.raises(ValueError, match="raw_time"):
        verify_sample({**shifted, "raw_time": shifted["raw_time"] + 0.001}, expected, 1, offset)


def test_legacy_import_preserves_policy_logits_and_values(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)

    def network(out):
        return nn.Sequential(
            nn.Linear(60, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, out)
        )

    actor, critic = network(16), network(1)
    state = {
        **{f"actor.{k}": v for k, v in actor.state_dict().items()},
        **{f"critic.{k}": v for k, v in critic.state_dict().items()},
    }
    checkpoint = tmp_path / "model.pt"
    torch.save(state, checkpoint)
    model = PPO(
        "MlpPolicy",
        ToyEnv(),
        n_steps=8,
        batch_size=8,
        policy_kwargs={"net_arch": dict(pi=[128, 128], vf=[128, 128]), "activation_fn": nn.Tanh},
    )
    import_legacy_weights(model, checkpoint)
    inputs = torch.rand(100, 60) * 2 - 1
    pi, vf = model.policy.mlp_extractor(inputs)
    torch.testing.assert_close(model.policy.action_net(pi), actor(inputs), rtol=0, atol=0)
    torch.testing.assert_close(model.policy.value_net(vf), critic(inputs), rtol=0, atol=0)
    legacy = LegacyPolicy(checkpoint)
    assert legacy(inputs[0].numpy()) == int(actor(inputs[0]).argmax())
    with pytest.raises(FileNotFoundError):
        load_policy(tmp_path / "absent.pt")
