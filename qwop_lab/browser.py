"""Headless browser adapter; no changes to game physics or installed package.

These top-level classes remain importable by Windows multiprocessing spawn.
The WebSocket transport never resends an ambiguous action after a timeout.
"""

import os
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from qwop_gym.envs.v1.util.wsserver import WSServer
from qwop_gym.envs.v1.util.wsclient import WSClient

BROWSER_FLAGS = [
    "--headless=new",
    "--allow-file-access-from-files",
    "--disable-extensions",
    "--disable-notifications",
    "--disable-popup-blocking",
    "--window-size=800,650",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--enable-unsafe-swiftshader",
    "--incognito",
]


class LabServer(WSServer):
    async def _launch_browser(self):
        options = webdriver.ChromeOptions()
        options.binary_location = self.browser
        for flag in BROWSER_FLAGS:
            options.add_argument(flag)
        self._driver = webdriver.Chrome(service=Service(self.driver), options=options)
        expected = os.environ.get("QWOP_LAB_BROWSER_VERSION")
        actual = self._driver.capabilities.get("browserVersion")
        if expected and actual != expected:
            self._driver.quit()
            raise RuntimeError(f"Browser changed: expected {expected}, got {actual}. Re-bootstrap.")
        self._window = self._driver.window_handles[0]
        self._driver.get(self.build_url())
        self._initialized = True


class StrictClient(WSClient):
    def connect(self):
        deadline = time.monotonic() + 35
        last_error = None
        while time.monotonic() < deadline:
            if self.shutdown.is_set():
                raise RuntimeError("Game process exited during connection")
            try:
                self._connect_attempt()
                return
            except Exception as exc:
                last_error = exc
                time.sleep(0.2)
        raise RuntimeError("Timed out connecting to QWOP") from last_error

    def send(self, data):
        try:
            self.ws.send(data)
            return self.ws.recv(timeout=30)
        except Exception as exc:
            raise RuntimeError(
                "Ambiguous game response; aborting without resending action"
            ) from exc
