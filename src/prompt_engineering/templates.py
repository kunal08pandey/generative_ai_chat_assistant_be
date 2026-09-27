class PromptTemplate:
    """
    A simple template class to build dynamic prompts.
    """
    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        """
        Formats the template using the provided keyword arguments.
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable for template: {e}")

    def __str__(self):
        return self.template
