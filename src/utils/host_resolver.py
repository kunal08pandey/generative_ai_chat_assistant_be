"""
Host Resolver Utility
======================

Dynamically resolves host IPs for Ollama and internal services.
Detects containerized vs. native execution environments so that
connections work seamlessly both inside Docker and natively on macOS/Linux.
Includes localhost verification for securing local-only models (e.g. Ollama).
"""
from __future__ import annotations

import os
import socket
import ipaddress
from typing import Any

LOCALHOST_HOSTS = {
    "127.0.0.1",
    "::1",
    "localhost",
    "testclient",
    "0.0.0.0",
}


def is_localhost(host: str | None) -> bool:
    """
    Check if a given host/IP string represents localhost / loopback.
    Handles IPv4 loopback (127.0.0.0/8), IPv6 loopback (::1),
    and IPv4-mapped IPv6 loopback (e.g. ::ffff:127.0.0.1).
    """
    if not host:
        return False
    normalized = host.strip().lower()
    if normalized in LOCALHOST_HOSTS:
        return True
    if normalized.startswith("127."):
        return True
    if normalized.startswith("::ffff:127.") or normalized == "::ffff:localhost":
        return True
    try:
        ip = ipaddress.ip_address(normalized)
        if ip.is_loopback:
            return True
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped and ip.ipv4_mapped.is_loopback:
            return True
    except ValueError:
        pass
    return False


def is_localhost_request(request: Any) -> bool:
    """
    Determine if an incoming HTTP request originates from localhost.
    Checks direct client host, X-Forwarded-For, and X-Real-IP headers.
    Supports ALLOW_REMOTE_OLLAMA=true for local network development.
    """
    if request is None:
        return True

    if os.environ.get("ALLOW_REMOTE_OLLAMA", "").strip().lower() in ("true", "1", "yes"):
        return True

    # 1. Direct client host
    client = getattr(request, "client", None)
    client_host = client.host if client else None

    # If no client host recorded (e.g., in-process test calls), allow
    if not client_host:
        return True

    if not is_localhost(client_host):
        return False

    # 2. Check proxy forwarded headers if present
    headers = getattr(request, "headers", {})
    forwarded_for = headers.get("x-forwarded-for")
    if forwarded_for:
        # Original client IP is the first in the chain
        first_hop = forwarded_for.split(",")[0].strip()
        if not is_localhost(first_hop):
            return False

    real_ip = headers.get("x-real-ip")
    if real_ip and not is_localhost(real_ip):
        return False

    return True


def get_ollama_host() -> str:
    """
    Return the appropriate host address for Ollama API calls.

    Ollama is restricted to localhost / container loopback.
    Priority:
      1. Explicit ``OLLAMA_HOST`` environment variable (if set and localhost/docker).
      2. ``host.docker.internal`` if running inside a Docker container.
      3. ``127.0.0.1`` if running natively on host machine.
    """
    if "OLLAMA_HOST" in os.environ and os.environ["OLLAMA_HOST"].strip():
        env_host = os.environ["OLLAMA_HOST"].strip()
        # If user explicitly set host.docker.internal but we're NOT in Docker and it can't resolve,
        # fallback to 127.0.0.1 to prevent instant DNS failure.
        if env_host == "host.docker.internal" and not os.path.exists("/.dockerenv"):
            try:
                socket.gethostbyname(env_host)
                return env_host
            except socket.gaierror:
                return "127.0.0.1"
        return env_host

    # Running inside Docker container
    if os.path.exists("/.dockerenv"):
        return "host.docker.internal"

    # Native execution strictly localhost
    return "127.0.0.1"
