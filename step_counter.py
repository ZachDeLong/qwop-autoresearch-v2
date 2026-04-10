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

import atexit
import json
import os

import gymnasium as gym
import qwop_gym  # noqa: F401

# === FROZEN CONSTANTS ===
STEP_BUDGET = 10_000_000
FRAMES_PER_STEP = 4
MAX_EPISODE_STEPS = 5000
SANITY_CHECK_BUDGET = 10_000  # free steps for verifying code works

# Chrome/driver paths — override via env vars for non-Windows systems
CHROME_PATH = os.environ.get(
    "QWOP_CHROME_PATH",
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
)
CHROMEDRIVER_PATH = os.environ.get(
    "QWOP_CHROMEDRIVER_PATH",
    os.path.expanduser("~/.cache/selenium/chromedriver/win64/145.0.7632.117/chromedriver.exe"),
)

# Persistent step count file
STEP_COUNT_FILE = "results/step_count.json"

# Checkpoint to disk every N steps (avoids per-step file I/O)
_CHECKPOINT_INTERVAL = 1000

# In-memory state (loaded once, checkpointed periodically)
_state: dict | None = None


def _ensure_loaded() -> dict:
    """Load state from disk on first access, then keep in memory."""
    global _state
    if _state is None:
        if os.path.exists(STEP_COUNT_FILE):
            with open(STEP_COUNT_FILE) as f:
                _state = json.load(f)
        else:
            _state = {"total_steps": 0, "sanity_steps": 0, "budget_active": False}
        _state.setdefault("_since_checkpoint", 0)
    return _state


def _save_to_disk() -> None:
    """Write current state to disk."""
    if _state is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(STEP_COUNT_FILE)), exist_ok=True)
    # Write a clean copy without internal bookkeeping
    disk_data = {k: v for k, v in _state.items() if not k.startswith("_")}
    with open(STEP_COUNT_FILE, "w") as f:
        json.dump(disk_data, f, indent=2)


def _maybe_checkpoint() -> None:
    """Save to disk if enough steps have passed since last checkpoint."""
    state = _ensure_loaded()
    state["_since_checkpoint"] += 1
    if state["_since_checkpoint"] >= _CHECKPOINT_INTERVAL:
        state["_since_checkpoint"] = 0
        _save_to_disk()


# Always flush to disk on exit so no steps are lost
atexit.register(_save_to_disk)


def get_step_count() -> int:
    """Return total budget steps consumed (excludes sanity check steps)."""
    return _ensure_loaded()["total_steps"]


def get_remaining_steps() -> int:
    """Return remaining budget steps."""
    return STEP_BUDGET - get_step_count()


def activate_budget():
    """Mark the budget as active (Phase 1 starts). Called before real training."""
    state = _ensure_loaded()
    state["budget_active"] = True
    _save_to_disk()


def is_budget_active() -> bool:
    return _ensure_loaded()["budget_active"]


class StepCounter(gym.Wrapper):
    """Gym wrapper that counts env.step() calls and enforces the budget."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        state = _ensure_loaded()

        if state["budget_active"]:
            if state["total_steps"] >= STEP_BUDGET:
                _save_to_disk()
                raise RuntimeError(
                    f"Step budget exhausted: {state['total_steps']}/{STEP_BUDGET} steps used. "
                    f"Go to evaluation with what you have."
                )
            state["total_steps"] += 1
        else:
            if state["sanity_steps"] >= SANITY_CHECK_BUDGET:
                _save_to_disk()
                raise RuntimeError(
                    f"Sanity check budget exhausted: {state['sanity_steps']}/{SANITY_CHECK_BUDGET}. "
                    f"Call activate_budget() to start using the main budget."
                )
            state["sanity_steps"] += 1

        _maybe_checkpoint()
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
    data = _ensure_loaded()
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
