"""
replay_renderer.py — FROZEN replay renderer for consistent video output.

Replays a saved action sequence at a fixed, real-time-ish speed so the
game is watchable. Used for screen recording after evaluation.

Usage:
  python replay_renderer.py --replay results/replays/baseline_best_100.2m_ep173.json
  python replay_renderer.py --replay results/replays/claude_best_107.2m_ep50.json --fps 15
"""

import argparse
import json
import os
import time

import gymnasium as gym
import qwop_gym  # noqa: F401

# === FROZEN CONSTANTS ===
FRAMES_PER_STEP = 4
DEFAULT_RENDER_FPS = 10  # steps per second for video playback

CHROME_PATH = os.environ.get(
    "QWOP_CHROME_PATH",
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
)
CHROMEDRIVER_PATH = os.environ.get(
    "QWOP_CHROMEDRIVER_PATH",
    os.path.expanduser("~/.cache/selenium/chromedriver/win64/145.0.7632.117/chromedriver.exe"),
)


def replay(replay_path: str, fps: int = DEFAULT_RENDER_FPS):
    """Load a replay JSON and play it back at the given FPS."""
    with open(replay_path) as f:
        data = json.load(f)

    actions = data["actions"]
    distance = data.get("distance", "?")
    game_time = data.get("game_time", "?")
    is_success = data.get("is_success", False)

    print(f"Replay: {os.path.basename(replay_path)}")
    print(f"  Distance:  {distance}m")
    print(f"  Game time: {game_time}s")
    print(f"  Success:   {is_success}")
    print(f"  Actions:   {len(actions)} steps")
    print(f"  Playback:  {fps} steps/sec")
    print()
    print("Starting in 3 seconds... (position your screen recorder)")
    time.sleep(3)

    env = gym.make(
        "QWOP-v1",
        browser=CHROME_PATH,
        driver=CHROMEDRIVER_PATH,
        frames_per_step=FRAMES_PER_STEP,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=5000)

    obs, _ = env.reset()
    delay = 1.0 / fps

    for i, action in enumerate(actions):
        step_start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            final_distance = info.get("distance", 0.0)
            final_time = float(info.get("time", 0.0)) * 10
            print(f"\nEpisode ended at step {i + 1}")
            print(f"  Distance:  {final_distance:.2f}m")
            print(f"  Game time: {final_time:.2f}s")
            break

        # Throttle to target FPS
        elapsed = time.time() - step_start
        if elapsed < delay:
            time.sleep(delay - elapsed)

    print("\nReplay complete. You can stop recording.")
    time.sleep(2)
    env.close()


def main():
    parser = argparse.ArgumentParser(description="QWOP Replay Renderer")
    parser.add_argument("--replay", required=True, help="Path to replay JSON file")
    parser.add_argument("--fps", type=int, default=DEFAULT_RENDER_FPS, help=f"Playback speed in steps/sec (default: {DEFAULT_RENDER_FPS})")
    args = parser.parse_args()

    replay(args.replay, args.fps)


if __name__ == "__main__":
    main()
