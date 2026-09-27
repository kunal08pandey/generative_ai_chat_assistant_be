import re
from src.utils.logger import get_logger

logger = get_logger(__name__)

def count_tokens(text: str) -> int:
    """
    A simple heuristic to estimate token count.
    In a real-world scenario, you might use 'tiktoken' for OpenAI 
    or just use the returned 'eval_count' from Ollama.
    This splits by words and punctuation, approximating 1 word ~ 1.3 tokens.
    """
    if not text:
        return 0
    words = re.findall(r'\w+|[^\w\s]', text)
    # Simple approx: words count + a small modifier
    estimated_count = int(len(words) * 1.1)
    
    logger.debug(f"Estimated token count: {estimated_count}")
    return estimated_count
