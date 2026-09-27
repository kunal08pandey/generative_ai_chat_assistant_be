from src.llm.base import BaseLLM
from src.prompt_engineering.templates import PromptTemplate
from src.utils.logger import get_logger

logger = get_logger(__name__)

class LLMChain:
    """
    Links a PromptTemplate with an LLM Client to execute a single generation step.
    """
    def __init__(self, prompt_template: PromptTemplate, llm: BaseLLM):
        self.prompt_template = prompt_template
        self.llm = llm

    async def run(self, **kwargs) -> str:
        prompt_text = self.prompt_template.format(**kwargs)
        logger.debug(f"LLMChain executing prompt:\n{prompt_text[:100]}...")
        result = await self.llm.generate(prompt_text)
        return result


class SequentialChain:
    """
    Links multiple LLMChains allowing the output of one step to be passed as an input variable to the next.
    """
    def __init__(self, chains: list, output_keys: list):
        if len(chains) != len(output_keys):
            raise ValueError("Number of chains must match number of output keys.")
        self.chains = chains
        self.output_keys = output_keys

    async def run(self, **kwargs) -> dict:
        current_context = kwargs.copy()
        for i, chain in enumerate(self.chains):
            logger.info(f"Executing chain step {i+1}/{len(self.chains)}")
            output = await chain.run(**current_context)
            out_key = self.output_keys[i]
            current_context[out_key] = output
            
        return current_context
