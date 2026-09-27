"""
Models Endpoint
================

``GET /models`` — list available LLM models grouped by provider.
"""
import httpx
from fastapi import APIRouter

from src.api.schemas import ok
from src.llm.registry import ModelRegistry
from src.settings import OLLAMA_BASE_URL
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/")
async def get_models():
    """
    Returns all available LLM models grouped by provider.

    Attempts to query the local Ollama instance dynamically.
    Falls back gracefully to default registered models if Ollama is starting up.
    """
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

    models_by_provider = ModelRegistry.get_models()
    all_models = ModelRegistry.get_all_models()
    providers = ModelRegistry.get_providers()

    logger.debug("Serving models list — %d providers, %d models", len(providers), len(all_models))
    return ok(data={
        "models_by_provider": models_by_provider,
        "all_models": all_models,
        "providers": providers,
    })
