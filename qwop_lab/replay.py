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


def ffmpeg_binary():
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("Install FFmpeg or the project's [video] extra") from exc


def clock_tolerance(actual_time, expected_time):
    # Each independently rounded float32 clock contributes at most half an ULP.
    # Above 64 seconds, one ULP exceeds the historical 5-microsecond floor.
    # The tiny extra term covers the separately rounded initial reset offset.
    rounding = (
        abs(float(np.spacing(np.float32(actual_time))))
        + abs(float(np.spacing(np.float32(expected_time))))
    ) / 2
    return max(5e-6, rounding + 1e-8)


def verify_sample(actual, expected, index, time_offset=0.0):
    # Rendering must not change any recorded physical transition.
    for key in ("obs_sha256", "distance", "terminated", "truncated", "is_success"):
        if actual[key] != expected[key]:
            raise ValueError(
                f"Replay diverged at step {index}: {key} "
                f"expected {expected[key]}, got {actual[key]}"
            )
    for key in ("gait_pose", "gait"):
        if key in expected and actual.get(key) != expected[key]:
            raise ValueError(f"Replay diverged at step {index}: {key}")
    if not math.isclose(
        actual["raw_time"] - expected["raw_time"],
        time_offset,
        rel_tol=0,
        abs_tol=clock_tolerance(actual["raw_time"], expected["raw_time"]),
    ):
        raise ValueError(f"Replay diverged at step {index}: raw_time after reset-offset correction")


def frame_repeats(previous_time, current_time, fps=30):
    if current_time <= previous_time:
        raise ValueError("Game clock did not advance")
    return round(current_time * fps) - round(previous_time * fps)


def annotated(rgb, label, state):
    im = Image.fromarray(rgb)
    header = 90 if "gait" in state else 64
    canvas = Image.new("RGB", (im.width, im.height + header), (18, 23, 32))
    canvas.paste(im, (0, header))
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
        f"{state['raw_time']:.2f}s game   {state['distance']:.2f}m   {status}",
        font=font,
        fill=(135, 217, 190),
    )
    if "gait" in state:
        gait = state["gait"]
        knee = min(gait["left_knee_clearance_m"], gait["right_knee_clearance_m"])
        draw.text(
            (12, 60),
            f"Hip {gait['pelvis_height_m']:.2f}m  Knee {knee:.2f}m  "
            f"Tilt {gait['torso_tilt_degrees']:.0f}deg  Upright {gait['upright']}",
            font=font,
            fill=(200, 210, 230),
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
    encoder = ffmpeg_binary()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial.mp4")
    case = Case(data["case"]["seed"], data["case"]["soft_resets"])
    lease = ledger.lease(f"replay:{label}")
    env = process = None
    frames = 0
    gait_samples = []
    max_clock_tolerance = 5e-6
    try:
        env = make_env(lease)
        if "gait_config" in data:
            from .gait import GaitConfig, GaitWrapper

            env = GaitWrapper(env, GaitConfig.from_dict(data["gait_config"]))
        obs, info = reset_case(env, case)
        state = sample(obs, info)
        time_offset = state["raw_time"] - data["initial"]["raw_time"]
        verify_sample(state, data["initial"], 0, time_offset)
        rgb = annotated(env.render(), label, state)
        height, width, _ = rgb.shape
        process = subprocess.Popen(
            [
                encoder,
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
            max_clock_tolerance = max(
                max_clock_tolerance, clock_tolerance(actual["raw_time"], expected["raw_time"])
            )
            count = frame_repeats(state["score_time"], actual["score_time"])
            process.stdin.write(rgb.tobytes() * count)
            frames += count
            state = actual
            rgb = annotated(env.render(), label, state)
            if "gait_config" in data:
                for fraction in (0.2, 0.5, 0.8):
                    if index == max(1, round(len(data["actions"]) * fraction)):
                        gait_samples.append((fraction, index, rgb.copy()))
        # One final state frame, followed by a clearly documented 1-second hold.
        process.stdin.write(rgb.tobytes() * 30)
        frames += 30
        process.stdin.close()
        error = process.stderr.read().decode(errors="replace")
        if process.wait(timeout=30) != 0:
            raise RuntimeError(f"FFmpeg failed: {error}")
        os.replace(partial, output)
        sample_meta = []
        for fraction, step, pixels in gait_samples:
            path = output.with_name(f"{output.stem}.sample-{round(fraction * 100)}.png")
            Image.fromarray(pixels).save(path)
            sample_meta.append(
                {"fraction": fraction, "step": step, "file": path.name, "sha256": sha256(path)}
            )
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
            "elapsed_raw_time_tolerance": max_clock_tolerance,
            "clock_tolerance_rule": "max(5e-6, half ULP(actual) + half ULP(expected) + 1e-8); float32 clocks",
            "final_score_time": state["score_time"],
            "overlay_clock": "raw info.time, game display scale; video uses simulation time",
            "case": data["case"],
            "contract_id": data["contract"]["contract_id"],
            **({"gait_samples": sample_meta} if sample_meta else {}),
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
            ffmpeg_binary(),
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
