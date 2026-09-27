"""
Host Resolver Utility
======================

Dynamically resolves host IPs for Ollama and internal services.
Detects containerized vs. native execution environments so that
connections work seamlessly both inside Docker and natively on macOS/Linux.
"""
import os
import socket


def get_ollama_host() -> str:
    """
    Return the appropriate host address for Ollama API calls.
    
    Priority:
      1. Explicit ``OLLAMA_HOST`` environment variable (if set).
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

    # Native execution
    return "127.0.0.1"
