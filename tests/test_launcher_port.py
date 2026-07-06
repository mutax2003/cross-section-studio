"""Tests for launcher port selection."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from launcher import DEFAULT_PORT, _is_port_free, _pick_port, _resolve_port


def test_is_port_free_on_unused_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        _, free_port = sock.getsockname()
    assert _is_port_free(free_port) is True


def test_is_port_false_when_port_occupied() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        _, port = sock.getsockname()
        assert _is_port_free(port) is False


def test_pick_port_returns_preferred_when_free() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        _, free_port = sock.getsockname()
    assert _pick_port(free_port, free_port) == free_port


def test_pick_port_advances_when_preferred_occupied() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        _, blocked_port = sock.getsockname()
        next_port = _pick_port(blocked_port, blocked_port + 5)
        assert next_port > blocked_port
        assert _is_port_free(next_port)


def test_resolve_port_uses_env_when_free(monkeypatch) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        _, free_port = sock.getsockname()
    monkeypatch.setenv("CROSS_SECTION_PORT", str(free_port))
    assert _resolve_port() == free_port


def test_resolve_port_falls_back_to_default_range(monkeypatch) -> None:
    monkeypatch.delenv("CROSS_SECTION_PORT", raising=False)
    port = _resolve_port()
    assert DEFAULT_PORT <= port <= DEFAULT_PORT + 20
