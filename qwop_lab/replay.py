"""Replay verifies the full state trace before accepting a video artifact."""

import os
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .artifacts import digest, read_json, sha256, write_json
from .environment import Case, contract, make_env, reset_case, runtime
from .evaluation import sample


def verify_sample(actual, expected, index, time_offset=0.0):
    # Rendering must not change any recorded physical transition.
    for key in ("obs_sha256", "distance", "terminated", "truncated", "is_success"):
        if actual[key] != expected[key]:
            raise ValueError(
                f"Replay diverged at step {index}: {key} "
                f"expected {expected[key]}, got {actual[key]}"
            )
    if not math.isclose(
        actual["raw_time"] - expected["raw_time"], time_offset, rel_tol=0, abs_tol=5e-6
    ):
        raise ValueError(f"Replay diverged at step {index}: raw_time after reset-offset correction")


def frame_repeats(previous_time, current_time, fps=30):
    if current_time <= previous_time:
        raise ValueError("Game clock did not advance")
    return round(current_time * fps) - round(previous_time * fps)


def annotated(rgb, label, state):
    im = Image.fromarray(rgb)
    canvas = Image.new("RGB", (im.width, im.height + 64), (18, 23, 32))
    canvas.paste(im, (0, 64))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 19)
    except OSError:
        font = ImageFont.load_default(size=19)
    draw.text((12, 5), label, font=font, fill="white")
    status = "FINISHED" if state["is_success"] else "RUNNING"
    if state["terminated"] and not state["is_success"]:
        status = "FELL"
    if state["truncated"]:
        status = "TIMEOUT"
    draw.text(
        (12, 33),
        f"{state['score_time']:.2f}s   {state['distance']:.2f}m   {status}",
        font=font,
        fill=(135, 217, 190),
    )
    return np.asarray(canvas)


def replay(replay_file, output, ledger, label="Verified replay"):
    data = read_json(replay_file)
    if data["contract"] != contract(runtime()):
        raise ValueError("Replay contract/runtime mismatch; do not compare incompatible runs")
    if (
        digest(data["actions"]) != data["actions_sha256"]
        or digest(data["trace"]) != data["trace_sha256"]
    ):
        raise ValueError("Replay artifact hash mismatch")
    if len(data["actions"]) != len(data["trace"]) or not data["actions"]:
        raise ValueError("Invalid replay action/trace lengths")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg must be installed and on PATH")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial.mp4")
    case = Case(data["case"]["seed"], data["case"]["soft_resets"])
    lease = ledger.lease(f"replay:{label}")
    env = process = None
    frames = 0
    try:
        env = make_env(lease)
        obs, info = reset_case(env, case)
        state = sample(obs, info)
        time_offset = state["raw_time"] - data["initial"]["raw_time"]
        verify_sample(state, data["initial"], 0, time_offset)
        rgb = annotated(env.render(), label, state)
        height, width, _ = rgb.shape
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                "30",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-movflags",
                "+faststart",
                str(partial),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for index, (action, expected) in enumerate(zip(data["actions"], data["trace"]), 1):
            obs, reward, terminated, truncated, info = env.step(action)
            actual = sample(obs, info, terminated, truncated, reward)
            verify_sample(actual, expected, index, time_offset)
            count = frame_repeats(state["score_time"], actual["score_time"])
            process.stdin.write(rgb.tobytes() * count)
            frames += count
            state = actual
            rgb = annotated(env.render(), label, state)
        # One final state frame, followed by a clearly documented 1-second hold.
        process.stdin.write(rgb.tobytes() * 30)
        frames += 30
        process.stdin.close()
        error = process.stderr.read().decode(errors="replace")
        if process.wait(timeout=30) != 0:
            raise RuntimeError(f"FFmpeg failed: {error}")
        os.replace(partial, output)
        meta = {
            "verified": True,
            "verified_transitions": len(data["actions"]),
            "source_replay_sha256": sha256(replay_file),
            "video_sha256": sha256(output),
            "fps": 30,
            "frames": frames,
            "duration_seconds": frames / 30,
            "final_hold_seconds": 1,
            "initial_score_time": data["initial"]["score_time"],
            "reset_raw_time_offset": time_offset,
            "elapsed_raw_time_tolerance": 5e-6,
            "final_score_time": state["score_time"],
            "case": data["case"],
            "contract_id": data["contract"]["contract_id"],
        }
        write_json(output.with_suffix(".json"), meta)
        Image.fromarray(rgb).save(output.with_suffix(".png"))
        return meta
    finally:
        if process and process.poll() is None:
            process.kill()
            process.wait()
        try:
            if env:
                env.close()
        finally:
            lease.close()
            if partial.exists():
                partial.unlink()


def side_by_side(left, right, output):
    metas = [read_json(Path(path).with_suffix(".json")) for path in (left, right)]
    if not all(m["verified"] for m in metas):
        raise ValueError("Only verified videos can be compared")
    if metas[0]["contract_id"] != metas[1]["contract_id"] or metas[0]["case"] != metas[1]["case"]:
        raise ValueError("Side-by-side videos require matching contracts and start cases")
    for path, meta in zip((left, right), metas):
        if sha256(path) != meta["video_sha256"]:
            raise ValueError("Video has changed since verification")
    duration = max(m["duration_seconds"] for m in metas)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-i",
            str(left),
            "-i",
            str(right),
            "-filter_complex",
            f"[0:v]tpad=stop_mode=clone:stop_duration={duration}[a];"
            f"[1:v]tpad=stop_mode=clone:stop_duration={duration}[b];[a][b]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    return {"path": str(output), "duration_seconds": duration}
