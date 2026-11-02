"""
Models Endpoint
================

``GET /models`` — list available LLM models grouped by provider.
``GET /models/groq`` — list Groq models categorized by organization.

Ensures that Ollama models are strictly accessible and advertised only on localhost.
"""
import httpx
from fastapi import APIRouter, Request

from src.api.schemas import ok
from src.llm.groq_client import GroqClient
from src.llm.registry import ModelRegistry
from src.settings import OLLAMA_BASE_URL
from src.utils.host_resolver import is_localhost_request
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/")
async def get_models(request: Request):
    """
    Returns all available LLM models grouped by provider.

    Security & Topology rule:
      • Ollama models are ONLY available on localhost.
      • If the incoming request originates from a non-localhost address,
        Ollama models are omitted from the catalog.
    """
    is_local = is_localhost_request(request)

    if is_local:
        # Dynamically query local Ollama instance on localhost
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                if response.status_code == 200:
                    installed_models = []
                    for item in response.json().get("models", []):
                        capabilities = item.get("capabilities") or []
                        model_name = item.get("name") or item.get("model")
                        if model_name and (not capabilities or "completion" in capabilities):
                            installed_models.append(model_name)
                    if installed_models:
                        ModelRegistry.set_models("ollama", installed_models)
        except Exception as exc:
            logger.warning("Could not discover Ollama models at %s: %s. Using default models.", OLLAMA_BASE_URL, exc)
    else:
        logger.info("Non-localhost request to /models (client: %s) — omitting Ollama models.", getattr(request.client, "host", "unknown"))

    models_by_provider = ModelRegistry.get_models(include_ollama=is_local)
    all_models = ModelRegistry.get_all_models(include_ollama=is_local)
    providers = ModelRegistry.get_providers(include_ollama=is_local)

    # Groq categorized model breakdown
    groq_categorized = GroqClient.list_models_categorized()

    logger.debug("Serving models list — %d providers, %d models (localhost=%s)", len(providers), len(all_models), is_local)
    return ok(data={
        "models_by_provider": models_by_provider,
        "all_models": all_models,
        "providers": providers,
        "groq_categories": groq_categorized,
        "is_localhost": is_local,
    })


@router.get("/groq")
async def get_groq_models():
    """
    Returns all available Groq models categorized by organization
    (Alibaba Cloud, Canopy Labs, Meta, OpenAI, Mistral AI, Google, DeepSeek).
    """
    categorized = GroqClient.list_models_categorized()
    flat = GroqClient.list_models()
    return ok(data={
        "categories": categorized,
        "all_models": flat,
    })
