"""Headless browser adapter; no changes to game physics or installed package.

These top-level classes remain importable by Windows multiprocessing spawn.
The WebSocket transport never resends an ambiguous action after a timeout.
"""

import os
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlsplit

import qwop_gym

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from qwop_gym.envs.v1.util.wsserver import WSServer
from qwop_gym.envs.v1.util.wsclient import WSClient
from qwop_gym.envs.v1.util.wsproto import WSProto, to_bytes
from websockets.sync.client import connect as websocket_connect

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


class GameFiles(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def list_directory(self, path):
        self.send_error(403, "Directory listing disabled")
        return None


class LabServer(WSServer):
    def build_url(self):
        # A loopback server lets Chrome load the local game on macOS without
        # requiring filesystem access to the user's Documents directory.
        if getattr(self, "_http_server", None) is None:
            game = Path(qwop_gym.__file__).parent / "envs/v1/game"
            self._http_server = ThreadingHTTPServer(
                ("127.0.0.1", 0), partial(GameFiles, directory=str(game))
            )
            self._http_thread = Thread(target=self._http_server.serve_forever, daemon=True)
            self._http_thread.start()
        query = urlsplit(super().build_url()).query
        return f"http://127.0.0.1:{self._http_server.server_port}/QWOP.html?{query}"

    def _close_http(self):
        server = getattr(self, "_http_server", None)
        if server is not None:
            self._http_server = None
            server.shutdown()
            server.server_close()
            self._http_thread.join(timeout=2)

    def cleanup_and_exit(self):
        # Close Chrome before awaiting the websocket server's shutdown. Waiting
        # for the browser websocket first can exceed the parent process timeout.
        driver, self._driver = self._driver, None
        try:
            if driver:
                driver.quit()
        finally:
            try:
                self._close_http()
            finally:
                super().cleanup_and_exit()

    def start(self, shutdown):
        try:
            super().start(shutdown)
        finally:
            # The upstream exception path can leave Chrome and its driver alive.
            if self._driver:
                try:
                    self._driver.quit()
                except Exception:
                    pass
            self._close_http()

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
    def _connect_attempt(self):
        self.ws = websocket_connect(f"ws://localhost:{self.port}", open_timeout=5, close_timeout=2)
        try:
            self.ws.send(to_bytes(WSProto.H_REG) + to_bytes(WSProto.REG_PY))
            data = self.ws.recv(timeout=5)
            if not data or data[0] != WSProto.H_ACK:
                raise RuntimeError("Invalid game registration response")
        except Exception:
            self.ws.close()
            raise

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
