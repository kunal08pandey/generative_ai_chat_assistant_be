from abc import ABC, abstractmethod
from typing import AsyncIterator


class BaseLLM(ABC):
    @abstractmethod
    async def generate(self, prompt: str, images: list[str] = None) -> str:
        """
        Generates a full response from the LLM.
        :param prompt: The text prompt.
        :param images: A list of base64 encoded strings for vision models.
        """
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str, images: list[str] = None) -> AsyncIterator[str]:
        """
        Streams response tokens from the LLM one chunk at a time.
        :param prompt: The text prompt.
        :param images: A list of base64 encoded strings for vision models.
        :yields: Individual text chunks as they arrive.
        """
        pass
