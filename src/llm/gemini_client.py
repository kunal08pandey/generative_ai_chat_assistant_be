from src.llm.base import BaseLLM
from typing import AsyncIterator
import google.generativeai as genai
import os

class GeminiClient(BaseLLM):
    def __init__(self, model: str):
        self.model_name = model
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(self.model_name)

    async def generate(self, prompt: str, images: list[str] = None) -> str:
        response = await self.model.generate_content_async(prompt)
        return response.text

    async def generate_stream(self, prompt: str, images: list[str] = None) -> AsyncIterator[str]:
        response = await self.model.generate_content_async(prompt, stream=True)
        async for chunk in response:
            if chunk.text:
                yield chunk.text
