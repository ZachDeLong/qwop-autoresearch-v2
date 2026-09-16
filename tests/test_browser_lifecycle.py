from unittest.mock import Mock

import pytest

from qwop_lab import browser, environment


def test_environment_shuts_down_server_even_when_client_close_fails():
    env = object.__new__(environment.LabEnv)
    env.client = Mock()
    env.client.close.side_effect = RuntimeError("socket already closed")
    env.shutdown = Mock()
    env.proc = Mock()
    env.proc.is_alive.side_effect = [True, False]
    with pytest.raises(RuntimeError, match="socket already closed"):
        env.close()
    env.shutdown.set.assert_called_once()
    env.proc.join.assert_called_once_with(timeout=15)
    env.proc.terminate.assert_not_called()


def test_failed_browser_server_quits_driver(monkeypatch):
    server = object.__new__(browser.LabServer)
    server._driver = Mock()
    monkeypatch.setattr(browser.WSServer, "start", Mock(side_effect=TimeoutError("startup")))
    with pytest.raises(TimeoutError):
        server.start(Mock())
    server._driver.quit.assert_called_once()


def test_registration_timeout_closes_connection_without_resending_game_action(monkeypatch):
    client = object.__new__(browser.StrictClient)
    client.port = 12345
    socket = Mock()
    socket.recv.side_effect = TimeoutError("registration")
    monkeypatch.setattr(browser, "websocket_connect", Mock(return_value=socket))
    with pytest.raises(TimeoutError):
        client._connect_attempt()
    socket.send.assert_called_once()
    socket.recv.assert_called_once_with(timeout=5)
    socket.close.assert_called_once()


def test_shutdown_closes_browser_before_waiting_for_websockets(monkeypatch):
    server = object.__new__(browser.LabServer)
    events = []
    driver = Mock()
    driver.quit.side_effect = lambda: events.append("browser closed")
    server._driver = driver
    monkeypatch.setattr(
        browser.WSServer, "cleanup_and_exit", lambda self: events.append("server stopped")
    )
    server.cleanup_and_exit()
    assert events == ["browser closed", "server stopped"]
    assert server._driver is None


def test_loopback_game_server_serves_assets_and_closes(monkeypatch):
    from urllib.request import urlopen
    from urllib.error import HTTPError

    server = object.__new__(browser.LabServer)
    monkeypatch.setattr(
        browser.WSServer, "build_url", lambda self: "file:///game/QWOP.html?seed=42"
    )
    try:
        url = server.build_url()
        assert url.startswith("http://127.0.0.1:") and url.endswith("?seed=42")
        with urlopen(url, timeout=2) as response:
            assert response.status == 200
            assert b"extensions.js" in response.read()
        with pytest.raises(HTTPError) as error:
            urlopen(url.split("/QWOP.html")[0] + "/assets/", timeout=2)
        assert error.value.code == 403
    finally:
        server._close_http()
    assert not server._http_thread.is_alive()
    server._close_http()


@pytest.mark.skipif(environment.os.name != "nt", reason="Windows process-tree fallback")
def test_shutdown_timeout_targets_only_owned_process_tree(monkeypatch):
    env = object.__new__(environment.LabEnv)
    env.client, env.shutdown, env.proc = Mock(), Mock(), Mock()
    env.proc.pid = 12345
    env.proc.is_alive.return_value = True
    runner = Mock()
    monkeypatch.setattr(environment.subprocess, "run", runner)
    env.close()
    runner.assert_called_once_with(
        ["taskkill", "/PID", "12345", "/T", "/F"], capture_output=True, timeout=10, check=False
    )
