"""Atomic artifacts and explicit source/runtime provenance."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def observation_hash(obs):
    return hashlib.sha256(obs.astype("<f4").tobytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def provenance():
    def git(*args):
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    sources = {
        str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
        for p in sorted((ROOT / "qwop_lab").glob("*.py"))
    }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "source_hashes": sources,
        "source_digest": digest(sources),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "torch",
                "gymnasium",
                "qwop-gym",
                "stable-baselines3",
                "sb3-contrib",
                "selenium",
                "websockets",
                "pillow",
            )
        },
    }
