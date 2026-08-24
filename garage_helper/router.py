"""
LLM Router — the single entry-point for every LLM call in this repo.

Usage:
    from garage_helper import LLMRouter

    router = LLMRouter(verbose=True)
    reply  = router.generate("What is attention?")               # returns plain string
    resp   = router.generate_response("Summarise this")          # returns LLMResponse with token counts
    reply  = router.generate("Summarise this", model="gpt-4o")   # per-call model override

GPT provider switching (Azure vs direct OpenAI):
    Set OPENAI_PROVIDER=azure   → always use AzureOpenAI (needs OPENAI_ENDPOINT)
    Set OPENAI_PROVIDER=openai  → always use direct OpenAI (needs OPENAI_API_KEY)
    Neither set                 → auto-detect: Azure if OPENAI_ENDPOINT is present, else direct OpenAI

Local / alternative providers:
    Ollama   — model="ollama/llama3.2"  (set OLLAMA_BASE_URL if not localhost:11434)
    DeepSeek — model="deepseek-chat"    (set DEEPSEEK_API_KEY)
"""

from __future__ import annotations

import os
from typing import Any

from .logging_config import get_logger
from .providers import (
    AzureOpenAIProvider,
    OpenAIProvider,
    OpenAICompatibleProvider,
    OllamaProvider,
    DeepSeekProvider,
    BedrockClaudeProvider,
    AnthropicProvider,
    GeminiProvider,
    LLMResponse,
)

# ---------------------------------------------------------------------------
# Model → provider mapping
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, str] = {
    # OpenAI / Azure  (resolved at call time via OPENAI_PROVIDER or endpoint detection)
    "gpt-4o":                 "auto_openai",
    "gpt-4o-mini":            "auto_openai",
    "gpt-4-turbo":            "auto_openai",
    "gpt-4":                  "auto_openai",
    "gpt-3.5-turbo":          "auto_openai",
    "o3":                     "auto_openai",
    "o3-mini":                "auto_openai",
    "o1":                     "auto_openai",
    "o1-mini":                "auto_openai",
    # Claude via Bedrock — full model IDs
    "anthropic.claude-opus-4-8-20251101-v1:0":   "bedrock_claude",
    "anthropic.claude-sonnet-5-20251029-v1:0":   "bedrock_claude",
    "anthropic.claude-haiku-4-5-20251001-v1:0":  "bedrock_claude",
    "anthropic.claude-3-5-sonnet-20241022-v2:0": "bedrock_claude",
    "anthropic.claude-3-5-haiku-20241022-v1:0":  "bedrock_claude",
    "anthropic.claude-3-opus-20240229-v1:0":     "bedrock_claude",
    # Claude via Anthropic direct API
    "claude-opus-4-8":   "anthropic",
    "claude-sonnet-5":   "anthropic",
    "claude-haiku-4-5":  "anthropic",
    "claude-3-5-sonnet": "anthropic",
    "claude-3-opus":     "anthropic",
    # Gemini
    "gemini-2.5-pro":   "gemini",
    "gemini-2.5-flash": "gemini",
    "gemini-2.0-flash": "gemini",
    "gemini-1.5-pro":   "gemini",
    "gemini-1.5-flash": "gemini",
    # DeepSeek
    "deepseek-chat":     "deepseek",
    "deepseek-reasoner": "deepseek",
    # Ollama — common models; any "ollama/..." prefix also works via best-effort guessing
    "ollama/llama3.2":   "ollama",
    "ollama/llama3.1":   "ollama",
    "ollama/mistral":    "ollama",
    "ollama/codellama":  "ollama",
    "ollama/phi3":       "ollama",
    "ollama/gemma2":     "ollama",
}

_PROVIDER_MAP = {
    "azure_openai":   AzureOpenAIProvider,
    "openai":         OpenAIProvider,
    "ollama":         OllamaProvider,
    "deepseek":       DeepSeekProvider,
    "bedrock_claude": BedrockClaudeProvider,
    "anthropic":      AnthropicProvider,
    "gemini":         GeminiProvider,
}

# Default model: override via DEFAULT_LLM_MODEL env var or the default_model param.
_DEFAULT_MODEL_ENV = "DEFAULT_LLM_MODEL"

# The repo default. If you route through a Bedrock application inference
# profile, set BEDROCK_INFERENCE_PROFILE_ARN in your environment — it's
# account-specific so it never belongs in source. Otherwise this falls back
# to a plain Bedrock Claude model ID.
# Override either way with DEFAULT_LLM_MODEL env var or the default_model param.
_FALLBACK_MODEL = (
    os.environ.get("BEDROCK_INFERENCE_PROFILE_ARN")
    or "anthropic.claude-3-5-haiku-20241022-v1:0"
)


class LLMRouter:
    """
    Route any LLM call to the right provider based on the model name.

    Parameters
    ----------
    default_model : str, optional
        Model used when no model is specified per-call.  Falls back to the
        DEFAULT_LLM_MODEL env var, then to Claude on Bedrock.
    verbose : bool
        When True, every request/response is logged to stdout (Jupyter-safe).
    """

    def __init__(self, default_model: str | None = None, verbose: bool = False):
        self.verbose = verbose
        self.log = get_logger(verbose)

        self.default_model = (
            default_model
            or os.environ.get(_DEFAULT_MODEL_ENV)
            or _FALLBACK_MODEL
        )
        self.log.debug("LLMRouter ready  default_model=%s", self.default_model)

        # provider instance cache — one per (model, provider_key) pair
        self._cache: dict[tuple[str, str], Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a prompt, return the reply as a plain string."""
        return self.generate_response(prompt, model=model, system=system, **kwargs).text

    def generate_response(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a prompt, return a full LLMResponse (text + token counts + stop_reason)."""
        model = model or self.default_model
        provider = self._get_provider(model)
        self.log.debug("generate()  model=%s  provider=%s  prompt_len=%d",
                       model, provider.name, len(prompt))
        return provider.generate(prompt, system=system, **kwargs)

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Multi-turn chat. Pass a list of {"role": ..., "content": ...} dicts.
        Flattens history into the prompt for providers that don't support native multi-turn.
        """
        model = model or self.default_model
        provider = self._get_provider(model)
        self.log.debug("chat()  model=%s  turns=%d", model, len(messages))

        history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in messages[:-1]
        )
        last = messages[-1]["content"]
        combined = f"{history}\n\nUser: {last}" if history else last
        return provider.generate(combined, **kwargs).text

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        return sorted(_MODEL_REGISTRY.keys())

    def provider_for(self, model: str) -> str:
        return self._resolve_provider_key(model)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_provider(self, model: str) -> Any:
        provider_key = self._resolve_provider_key(model)
        cache_key = (model, provider_key)
        if cache_key not in self._cache:
            cls = _PROVIDER_MAP[provider_key]
            self._cache[cache_key] = cls(model, verbose=self.verbose)
            self.log.debug("Instantiated  %s  for model=%s", cls.__name__, model)
        return self._cache[cache_key]

    def _resolve_provider_key(self, model: str) -> str:
        provider_key = _MODEL_REGISTRY.get(model)

        # Prefix / substring match for long Bedrock ARN-style IDs
        if provider_key is None:
            for registered, pkey in _MODEL_REGISTRY.items():
                if model.startswith(registered) or registered.startswith(model):
                    provider_key = pkey
                    break

        # Best-effort guess from model name conventions
        if provider_key is None:
            low = model.lower()
            if low.startswith("arn:aws:bedrock"):
                provider_key = "bedrock_claude"
            elif "claude" in low and ("anthropic." in low or low.startswith("us.")):
                provider_key = "bedrock_claude"
            elif "claude" in low:
                provider_key = "anthropic"
            elif "gemini" in low:
                provider_key = "gemini"
            elif low.startswith("ollama/") or low.startswith("ollama:"):
                provider_key = "ollama"
            elif "deepseek" in low:
                provider_key = "deepseek"
            else:
                provider_key = "auto_openai"
            self.log.debug("Unknown model '%s' — guessed provider='%s'", model, provider_key)

        # Resolve auto_openai using explicit OPENAI_PROVIDER override first,
        # then fall back to endpoint detection.
        #
        # OPENAI_PROVIDER=azure   → AzureOpenAI (requires OPENAI_ENDPOINT)
        # OPENAI_PROVIDER=openai  → direct OpenAI (requires OPENAI_API_KEY)
        # (unset)                 → Azure if OPENAI_ENDPOINT present, else direct OpenAI
        if provider_key == "auto_openai":
            explicit = os.environ.get("OPENAI_PROVIDER", "").lower().strip()
            if explicit == "azure":
                provider_key = "azure_openai"
            elif explicit == "openai":
                provider_key = "openai"
            elif os.environ.get("OPENAI_ENDPOINT") or os.environ.get("AZURE_OPENAI_ENDPOINT"):
                provider_key = "azure_openai"
            else:
                provider_key = "openai"
            self.log.debug("auto_openai → %s  (OPENAI_PROVIDER=%r)", provider_key, explicit or "(unset)")

        return provider_key
