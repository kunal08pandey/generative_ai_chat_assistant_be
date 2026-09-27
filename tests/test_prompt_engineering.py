import asyncio

import pytest

from src.prompt_engineering.chainer import LLMChain, SequentialChain
from src.prompt_engineering.few_shot import FewShotPromptTemplate
from src.prompt_engineering.templates import PromptTemplate


class FakeLLM:
    async def generate(self, prompt, images=None):
        return f"response:{prompt}"


def test_prompt_templates():
    assert PromptTemplate("Hello {name}").format(name="Ada") == "Hello Ada"
    with pytest.raises(ValueError):
        PromptTemplate("Hello {name}").format()

    few_shot = FewShotPromptTemplate(
        prefix="System",
        examples=[{"input": "Hi", "output": "Hello"}],
        suffix="User: {query}\nAssistant:",
    )
    assert "User: Hi\nAssistant: Hello" in few_shot.format(query="Test")


def test_async_chains():
    first = LLMChain(PromptTemplate("first:{input}"), FakeLLM())
    second = LLMChain(PromptTemplate("second:{first_output}"), FakeLLM())
    chain = SequentialChain([first, second], ["first_output", "second_output"])
    result = asyncio.run(chain.run(input="value"))
    assert result["second_output"].startswith("response:second:")
