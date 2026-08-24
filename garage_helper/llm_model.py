# Kept for backwards compatibility. New code should use LLMRouter directly.
from .router import LLMRouter

def get_llm(model_name: str, verbose: bool = False) -> LLMRouter:
    return LLMRouter(default_model=model_name, verbose=verbose)
