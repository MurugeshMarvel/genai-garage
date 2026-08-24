# Kept for backwards compatibility. New code should use LLMRouter directly.
from .router import LLMRouter


def get_llm_response(prompt: str, model: str | None = None, verbose: bool = False) -> str:
    router = LLMRouter(verbose=verbose)
    return router.generate(prompt, model=model)
