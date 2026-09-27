from typing import List, Dict

class FewShotPromptTemplate:
    """
    Builds a prompt from a set of few-shot examples and a prefix/suffix.
    """
    def __init__(self, prefix: str, examples: List[Dict[str, str]], suffix: str, example_separator: str = "\n\n"):
        self.prefix = prefix
        self.examples = examples
        self.suffix = suffix
        self.example_separator = example_separator

    def format(self, **kwargs) -> str:
        """
        Builds the complete string: prefix + examples + formatted suffix.
        """
        formatted_parts = [self.prefix]
        
        for example in self.examples:
            # Assuming examples have "input" and "output" structure
            input_text = example.get("input", "")
            output_text = example.get("output", "")
            formatted_parts.append(f"User: {input_text}\nAssistant: {output_text}")
            
        try:
            formatted_suffix = self.suffix.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable for suffix template: {e}")
            
        formatted_parts.append(formatted_suffix)
        
        return self.example_separator.join(formatted_parts)
