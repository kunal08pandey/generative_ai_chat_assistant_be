"""
LLM Error Parser & Explainer Utility
====================================

Inspects backend and provider exceptions (Gemini, Ollama, Llama.cpp) and returns
a structured, human-readable error explanation with root causes and actionable suggestions.
"""
from __future__ import annotations

import json
from typing import Any, Dict


def parse_llm_error(exc: Exception | str, model: str = "") -> Dict[str, Any]:
    """
    Parse an exception into a clear, structured error payload:
      - code: Machine-readable category code
      - title: Short friendly title
      - message: User-friendly explanation of what happened
      - reason: The root cause of the error
      - suggestion: Step-by-step guidance to resolve it
      - technical: Raw error message for troubleshooting
      - model: Target model name
    """
    raw_str = str(exc)
    err_lower = raw_str.lower()
    clean_model = model or "Selected Model"

    # 1. Ollama Connection Issues
    if any(k in err_lower for k in [
        "all connection attempts failed",
        "connection refused",
        "connecterror",
        "failed to connect to host.docker.internal",
        "nodename nor servname provided",
    ]):
        return {
            "code": "OLLAMA_CONNECTION_ERROR",
            "title": "Cannot Connect to Local Ollama",
            "message": f"The backend could not reach the Ollama service on port 11434.",
            "reason": (
                "Ollama is either not running on your host machine, or is listening only on "
                "127.0.0.1 (which Docker containers cannot access via host.docker.internal)."
            ),
            "suggestion": (
                "1. If running locally, start Ollama with: OLLAMA_HOST=0.0.0.0:11434 ollama serve\n"
                "2. Or switch to a Google Gemini model (e.g. 'gemini-3.5-flash') using the model dropdown above."
            ),
            "model": clean_model,
            "technical": raw_str[:300],
        }

    # 2. Gemini 429 Quota Exceeded / Free Tier Limit 0
    if any(k in err_lower for k in ["429", "resource_exhausted", "quota exceeded", "quota failure"]):
        is_limit_zero = "limit: 0" in err_lower or "limit 0" in err_lower
        return {
            "code": "QUOTA_EXCEEDED",
            "title": "Google Quota Exceeded (429)",
            "message": f"Quota limit reached for model '{clean_model}'.",
            "reason": (
                f"Google AI Studio enforces a quota limit of 0 for '{clean_model}' on the Free Tier plan. "
                "Frontier reasoning and preview models require an active Pay-As-You-Go billing account."
                if is_limit_zero
                else "You have temporarily exceeded the requests per minute or daily token limit for this model."
            ),
            "suggestion": (
                "1. Switch to a free-tier supported model like 'gemini-3.5-flash' or 'gemini-3.5-flash-lite' from the dropdown.\n"
                "2. Or attach a billing account to your project in Google AI Studio to unlock frontier models."
            ),
            "model": clean_model,
            "technical": raw_str[:300],
        }

    # 3. Gemini 404 Model Not Found / Deprecated
    if any(k in err_lower for k in ["404", "not found", "no longer available", "does not exist"]):
        return {
            "code": "MODEL_NOT_FOUND",
            "title": "Model Not Found or Deprecated (404)",
            "message": f"Model '{clean_model}' is not available or has been retired by the provider.",
            "reason": (
                f"Google has retired '{clean_model}' for new API keys or the model identifier is deprecated."
            ),
            "suggestion": (
                "Please switch to 'gemini-3.5-flash' or 'gemini-3.1-pro-preview' from the dropdown menu above."
            ),
            "model": clean_model,
            "technical": raw_str[:300],
        }

    # 4. Gemini 503 Overloaded
    if any(k in err_lower for k in ["503", "unavailable", "high demand", "overloaded", "server is busy"]):
        alt_model = "gemini-3.5-flash-lite" if "flash-lite" not in clean_model.lower() else "gemini-3.5-flash"
        return {
            "code": "SERVICE_OVERLOADED",
            "title": "Provider Temporarily Overloaded (503)",
            "message": f"The upstream servers for '{clean_model}' are currently experiencing high demand.",
            "reason": "Google's model servers are temporarily experiencing high traffic load or brief downtime.",
            "suggestion": (
                f"1. Spikes in demand are usually temporary — wait 5-10 seconds and retry.\n"
                f"2. Or switch to an alternate model like '{alt_model}' from the dropdown above."
            ),
            "model": clean_model,
            "technical": raw_str[:300],
        }

    # 5. Auth / API Key Error
    if any(k in err_lower for k in ["401", "403", "unauthorized", "permission_denied", "api_key_invalid", "no api key provided"]):
        return {
            "code": "AUTH_ERROR",
            "title": "API Authentication Error",
            "message": "Failed to authenticate with the LLM API provider.",
            "reason": "The GEMINI_API_KEY is either missing, invalid, or lacks permission for this project.",
            "suggestion": (
                "Check your backend/.env file to ensure GEMINI_API_KEY is set correctly, then restart the server."
            ),
            "model": clean_model,
            "technical": raw_str[:300],
        }

    # 6. Timeout
    if any(k in err_lower for k in ["timeout", "timed out", "deadline exceeded"]):
        return {
            "code": "TIMEOUT_ERROR",
            "title": "Request Timed Out",
            "message": f"The request to '{clean_model}' took too long to respond.",
            "reason": "The generation query took longer than the 120-second connection timeout.",
            "suggestion": (
                "Try submitting a shorter prompt, reducing attached document size, or using a faster model."
            ),
            "model": clean_model,
            "technical": raw_str[:300],
        }

    # 7. Generic Fallback
    return {
        "code": "LLM_ERROR",
        "title": "Model Execution Error",
        "message": f"An error occurred while generating a response with '{clean_model}'.",
        "reason": raw_str[:160] if len(raw_str) > 5 else "An unexpected exception was encountered.",
        "suggestion": "Check backend container logs for details or switch to another model.",
        "model": clean_model,
        "technical": raw_str[:300],
    }
