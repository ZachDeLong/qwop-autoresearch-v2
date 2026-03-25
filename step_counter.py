"""
step_counter.py — FROZEN step-counting wrapper for budget enforcement.

Wraps any qwop-gym env and tracks total env.step() calls across
the entire autoresearch session. Claude's code must use make_counted_env()
instead of raw gym.make() to ensure budget tracking.

Usage:
    from step_counter import make_counted_env, get_step_count, STEP_BUDGET

    env = make_counted_env()
    # ... train ...
    print(f"Steps used: {get_step_count()} / {STEP_BUDGET}")
"""

import json
import os

import gymnasium as gym
import qwop_gym  # noqa: F401

# === FROZEN CONSTANTS ===
STEP_BUDGET = 10_000_000
FRAMES_PER_STEP = 4
MAX_EPISODE_STEPS = 5000
SANITY_CHECK_BUDGET = 10_000  # free steps for verifying code works

# Chrome paths
CHROME_PATH = "C:/Program Files/Google/Chrome/Application/chrome.exe"
CHROMEDRIVER_PATH = os.path.expanduser(
    "~/.cache/selenium/chromedriver/win64/145.0.7632.117/chromedriver.exe"
)

# Persistent step count file
STEP_COUNT_FILE = "results/step_count.json"


def _load_step_count() -> dict:
    if os.path.exists(STEP_COUNT_FILE):
        with open(STEP_COUNT_FILE) as f:
            return json.load(f)
    return {"total_steps": 0, "sanity_steps": 0, "budget_active": False}


def _save_step_count(data: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(STEP_COUNT_FILE)), exist_ok=True)
    with open(STEP_COUNT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_step_count() -> int:
    """Return total budget steps consumed (excludes sanity check steps)."""
    return _load_step_count()["total_steps"]


def get_remaining_steps() -> int:
    """Return remaining budget steps."""
    return STEP_BUDGET - get_step_count()


def activate_budget():
    """Mark the budget as active (Phase 1 starts). Called before real training."""
    data = _load_step_count()
    data["budget_active"] = True
    _save_step_count(data)


def is_budget_active() -> bool:
    return _load_step_count()["budget_active"]


class StepCounter(gym.Wrapper):
    """Gym wrapper that counts env.step() calls and enforces the budget."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        data = _load_step_count()

        if data["budget_active"]:
            if data["total_steps"] >= STEP_BUDGET:
                raise RuntimeError(
                    f"Step budget exhausted: {data['total_steps']}/{STEP_BUDGET} steps used. "
                    f"Go to evaluation with what you have."
                )
            data["total_steps"] += 1
        else:
            if data["sanity_steps"] >= SANITY_CHECK_BUDGET:
                raise RuntimeError(
                    f"Sanity check budget exhausted: {data['sanity_steps']}/{SANITY_CHECK_BUDGET}. "
                    f"Call activate_budget() to start using the main budget."
                )
            data["sanity_steps"] += 1

        _save_step_count(data)
        return self.env.step(action)


def make_counted_env(**kwargs):
    """Create a QWOP env with step counting.

    Returns a raw env (no logging wrapper). Claude can add their own
    wrappers on top, but the StepCounter is always innermost.

    Accepts optional kwargs to pass to gym.make (e.g., render_mode).
    """
    env = gym.make(
        "QWOP-v1",
        browser=CHROME_PATH,
        driver=CHROMEDRIVER_PATH,
        frames_per_step=FRAMES_PER_STEP,
        **kwargs,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    env = StepCounter(env)
    return env


def print_budget_status():
    """Print current budget usage."""
    data = _load_step_count()
    print(f"\n{'='*40}")
    print(f"STEP BUDGET STATUS")
    print(f"{'='*40}")
    print(f"  Budget active:  {data['budget_active']}")
    print(f"  Sanity steps:   {data['sanity_steps']:,} / {SANITY_CHECK_BUDGET:,}")
    print(f"  Budget steps:   {data['total_steps']:,} / {STEP_BUDGET:,}")
    print(f"  Remaining:      {STEP_BUDGET - data['total_steps']:,}")
    pct = data['total_steps'] / STEP_BUDGET * 100
    print(f"  Used:           {pct:.1f}%")
    print(f"{'='*40}\n")
