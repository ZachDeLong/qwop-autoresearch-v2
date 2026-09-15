"""Acquire the original game locally and pin its patched runtime identity."""

import importlib.metadata
import urllib.request
from pathlib import Path

import qwop_gym
from qwop_gym.tools.patch import patch
from selenium.webdriver.common.selenium_manager import SeleniumManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

from .artifacts import ROOT, digest, read_json, sha256, write_json
from .browser import BROWSER_FLAGS

SOURCE_URL = "https://www.foddy.net/legacy/QWOP.min.js"


def bootstrap(browser=None):
    folder = ROOT / ".runtime"
    folder.mkdir(exist_ok=True)
    source = folder / "QWOP.original.js"
    if not source.exists():
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "qwop-lab/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            source.write_bytes(response.read())
    lockfile = ROOT / "game-source.lock.json"
    source_hash = sha256(source)
    if lockfile.exists() and read_json(lockfile)["source_sha256"] != source_hash:
        raise RuntimeError("Game source differs from the committed lock; inspect before upgrading")
    patch(str(source))
    arguments = ["--browser", "chrome"]
    if browser:
        arguments += ["--browser-path", str(Path(browser).resolve())]
    resolved = SeleniumManager().binary_paths(arguments)
    browser_path, driver_path = resolved["browser_path"], resolved["driver_path"]
    options = webdriver.ChromeOptions()
    options.binary_location = browser_path
    for flag in BROWSER_FLAGS:
        options.add_argument(flag)
    with webdriver.Chrome(service=Service(driver_path), options=options) as probe:
        version = probe.capabilities["browserVersion"]
    game_dir = Path(qwop_gym.__file__).parent / "envs" / "v1" / "game"
    files = {
        "browser": browser_path,
        "driver": driver_path,
        "game": game_dir / "QWOP.min.js",
        "extension": game_dir / "extensions.js",
        "websocket_js": game_dir / "ws.js",
    }
    entries = {
        key: {"path": str(Path(value).resolve()), "sha256": sha256(value)}
        for key, value in files.items()
    }
    if lockfile.exists() and read_json(lockfile)["patched_sha256"] != entries["game"]["sha256"]:
        raise RuntimeError("Patched game differs from the committed lock; inspect before upgrading")
    identity = {
        "file_hashes": {k: v["sha256"] for k, v in entries.items()},
        "browser_version": version,
        "browser_flags": BROWSER_FLAGS,
        "qwop_gym_version": importlib.metadata.version("qwop-gym"),
    }
    manifest = {**identity, "files": entries, "runtime_id": digest(identity)}
    write_json(folder / "runtime.json", manifest)
    write_json(
        lockfile,
        {
            "url": SOURCE_URL,
            "source_sha256": source_hash,
            "patched_sha256": entries["game"]["sha256"],
            "qwop_gym": "1.0.1",
        },
    )
    import_checkpoints()
    return manifest


def import_checkpoints():
    lock = read_json(ROOT / "checkpoints.lock.json")
    folder = ROOT / ".runtime" / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    for name, item in lock["models"].items():
        path = folder / name
        if not path.exists():
            url = (
                "https://raw.githubusercontent.com/ZachDeLong/qwop-autoresearch-v2/"
                f"{lock['commit']}/{item['source_path']}"
            )
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            import hashlib

            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise RuntimeError(f"Historical checkpoint hash mismatch: {name}")
            path.write_bytes(content)
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"Historical checkpoint changed: {path}")
