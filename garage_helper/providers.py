"""
One class per provider. Each class accepts a prompt string and returns an LLMResponse.
All heavy imports are deferred — you only pay for the packages you actually use.

Provider hierarchy:
    OpenAICompatibleProvider  ← generic base for any OpenAI-compatible endpoint
        OllamaProvider        ← local models via Ollama (http://localhost:11434/v1)
        DeepSeekProvider      ← DeepSeek API (https://api.deepseek.com/v1)
    AzureOpenAIProvider       ← Azure-hosted OpenAI models
    OpenAIProvider            ← direct OpenAI API
    BedrockClaudeProvider     ← Claude via AWS Bedrock
    AnthropicProvider         ← Claude via Anthropic direct API
    GeminiProvider            ← Google Gemini

Adding a new OpenAI-compatible provider takes ~5 lines:
    class GroqProvider(OpenAICompatibleProvider):
        name = "groq"
        def __init__(self, model, verbose=False):
            super().__init__(model, "https://api.groq.com/openai/v1",
                             api_key_env="GROQ_API_KEY", verbose=verbose)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .logging_config import get_logger

# Newer OpenAI models (o1, o3, gpt-5.x) dropped `max_tokens` in favour of
# `max_completion_tokens`. Older models don't accept the new name, so we detect
# by model prefix rather than trying one and catching the error.
_COMPLETION_TOKENS_PREFIXES = ("o1", "o3", "gpt-5")


def _max_tokens_key(model: str) -> str:
    low = model.lower().lstrip("/ ")
    if any(low.startswith(p) for p in _COMPLETION_TOKENS_PREFIXES):
        return "max_completion_tokens"
    return "max_tokens"


@dataclass
class LLMResponse:
    """Uniform response object returned by every provider and router.generate()."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = "end_turn"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# OpenAI-compatible base  (reused by Ollama, DeepSeek, Groq, vLLM, etc.)
# ---------------------------------------------------------------------------

class OpenAICompatibleProvider:
    """
    Generic provider for any server that speaks the OpenAI Chat Completions API.

    Subclass it and override `name`, then call super().__init__() with the
    correct base_url and api_key_env. That's all you need to add a new provider.
    """
    name = "openai_compatible"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str | None = None,
        api_key_default: str = "no-key",
        verbose: bool = False,
    ):
        from openai import OpenAI
        self.model = model
        self.log = get_logger(verbose)

        api_key = (os.environ.get(api_key_env) if api_key_env else None) or api_key_default
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.log.debug("%s ready — model=%s  base_url=%s", self.__class__.__name__, model, base_url)

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Remap max_tokens → max_completion_tokens for models that require it
        if "max_tokens" in kwargs:
            kwargs[_max_tokens_key(self.model)] = kwargs.pop("max_tokens")

        self.log.debug("→ %s  model=%s  prompt_len=%d", self.name, self.model, len(prompt))
        resp = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        text = resp.choices[0].message.content or ""

        finish = resp.choices[0].finish_reason or "stop"
        stop_reason = {"stop": "end_turn", "length": "max_tokens"}.get(finish, finish)

        self.log.debug("← %s  tokens=%s  reply_len=%d",
                       self.name, resp.usage.total_tokens if resp.usage else "?", len(text))
        return LLMResponse(
            text=text,
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            stop_reason=stop_reason,
        )


# ---------------------------------------------------------------------------
# Ollama  (local, OpenAI-compatible — no API key required)
# Env: OLLAMA_BASE_URL  (default: http://localhost:11434/v1)
# Model examples: "ollama/llama3.2", "llama3.2", "mistral", "codellama"
# ---------------------------------------------------------------------------

class OllamaProvider(OpenAICompatibleProvider):
    name = "ollama"

    def __init__(self, model: str, verbose: bool = False):
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        # Ollama strips the "ollama/" prefix — pass the raw model name
        clean_model = model.removeprefix("ollama/")
        super().__init__(clean_model, base_url=base_url, api_key_default="ollama", verbose=verbose)
        self.model = clean_model


# ---------------------------------------------------------------------------
# DeepSeek  (OpenAI-compatible, hosted API)
# Env: DEEPSEEK_API_KEY
# Model examples: "deepseek-chat", "deepseek-reasoner"
# ---------------------------------------------------------------------------

class DeepSeekProvider(OpenAICompatibleProvider):
    name = "deepseek"

    def __init__(self, model: str, verbose: bool = False):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError("DeepSeek requires DEEPSEEK_API_KEY to be set.")
        super().__init__(
            model,
            base_url="https://api.deepseek.com/v1",
            api_key_env="DEEPSEEK_API_KEY",
            verbose=verbose,
        )


# ---------------------------------------------------------------------------
# Azure OpenAI  (OPENAI_ENDPOINT + OPENAI_API_KEY)
# Set OPENAI_PROVIDER=azure to force this even when OPENAI_ENDPOINT is absent.
# ---------------------------------------------------------------------------

class AzureOpenAIProvider:
    name = "azure_openai"

    def __init__(self, model: str, verbose: bool = False):
        from openai import AzureOpenAI
        self.model = model
        self.log = get_logger(verbose)

        endpoint = os.environ.get("OPENAI_ENDPOINT") or os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key  = os.environ.get("OPENAI_API_KEY")  or os.environ.get("AZURE_OPENAI_API_KEY")

        if not endpoint:
            raise EnvironmentError(
                "Azure OpenAI requires OPENAI_ENDPOINT (or AZURE_OPENAI_ENDPOINT) to be set."
            )
        if not api_key:
            raise EnvironmentError(
                "Azure OpenAI requires OPENAI_API_KEY (or AZURE_OPENAI_API_KEY) to be set."
            )

        if "/openai" in endpoint:
            endpoint = endpoint.split("/openai")[0]

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-02-01",
        )
        self.log.debug("AzureOpenAIProvider ready — model=%s", model)

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if "max_tokens" in kwargs:
            kwargs[_max_tokens_key(self.model)] = kwargs.pop("max_tokens")

        self.log.debug("→ AzureOpenAI  model=%s  prompt_len=%d", self.model, len(prompt))
        resp = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        text = resp.choices[0].message.content or ""

        finish = resp.choices[0].finish_reason or "stop"
        stop_reason = {"stop": "end_turn", "length": "max_tokens"}.get(finish, finish)

        self.log.debug("← AzureOpenAI  tokens=%s  reply_len=%d",
                       resp.usage.total_tokens if resp.usage else "?", len(text))
        return LLMResponse(
            text=text,
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            stop_reason=stop_reason,
        )


# ---------------------------------------------------------------------------
# OpenAI direct  (OPENAI_API_KEY, no endpoint override)
# Set OPENAI_PROVIDER=openai to force this even when OPENAI_ENDPOINT is set.
# ---------------------------------------------------------------------------

class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str, verbose: bool = False):
        from openai import OpenAI
        self.model = model
        self.log = get_logger(verbose)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OpenAI requires OPENAI_API_KEY to be set.")

        self.client = OpenAI(api_key=api_key)
        self.log.debug("OpenAIProvider ready — model=%s", model)

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if "max_tokens" in kwargs:
            kwargs[_max_tokens_key(self.model)] = kwargs.pop("max_tokens")

        self.log.debug("→ OpenAI  model=%s  prompt_len=%d", self.model, len(prompt))
        resp = self.client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        text = resp.choices[0].message.content or ""

        finish = resp.choices[0].finish_reason or "stop"
        stop_reason = {"stop": "end_turn", "length": "max_tokens"}.get(finish, finish)

        self.log.debug("← OpenAI  tokens=%s  reply_len=%d",
                       resp.usage.total_tokens if resp.usage else "?", len(text))
        return LLMResponse(
            text=text,
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            stop_reason=stop_reason,
        )


# ---------------------------------------------------------------------------
# Anthropic (Claude) via AWS Bedrock  (AWS credentials via env or ~/.aws)
# ---------------------------------------------------------------------------

class BedrockClaudeProvider:
    name = "bedrock_claude"

    def __init__(self, model: str, verbose: bool = False):
        import boto3
        self.model = model
        self.log = get_logger(verbose)

        region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.log.debug("BedrockClaudeProvider ready — model=%s region=%s", model, region)

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> LLMResponse:
        import json

        max_tokens = kwargs.pop("max_tokens", 1024)
        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        body.update(kwargs)

        self.log.debug("→ Bedrock/Claude  model=%s  prompt_len=%d", self.model, len(prompt))
        resp = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(resp["body"].read())
        text    = result["content"][0]["text"]
        usage   = result.get("usage", {})

        self.log.debug("← Bedrock/Claude  in=%s out=%s  reply_len=%d",
                       usage.get("input_tokens", "?"), usage.get("output_tokens", "?"), len(text))
        return LLMResponse(
            text=text,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            stop_reason=result.get("stop_reason", "end_turn"),
        )


# ---------------------------------------------------------------------------
# Anthropic direct (ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, verbose: bool = False):
        import anthropic
        self.model = model
        self.log = get_logger(verbose)

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("Anthropic requires ANTHROPIC_API_KEY to be set.")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.log.debug("AnthropicProvider ready — model=%s", model)

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> LLMResponse:
        max_tokens = kwargs.pop("max_tokens", 1024)
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }
        if system:
            create_kwargs["system"] = system

        self.log.debug("→ Anthropic  model=%s  prompt_len=%d", self.model, len(prompt))
        resp = self.client.messages.create(**create_kwargs)
        text = resp.content[0].text

        self.log.debug("← Anthropic  in=%s out=%s  reply_len=%d",
                       resp.usage.input_tokens, resp.usage.output_tokens, len(text))
        return LLMResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            stop_reason=resp.stop_reason or "end_turn",
        )


# ---------------------------------------------------------------------------
# Google Gemini (GOOGLE_API_KEY)
# ---------------------------------------------------------------------------

class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, verbose: bool = False):
        import google.generativeai as genai
        self.model_name = model
        self.log = get_logger(verbose)

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("Gemini requires GOOGLE_API_KEY to be set.")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.log.debug("GeminiProvider ready — model=%s", model)

    def generate(self, prompt: str, system: str | None = None, **kwargs: Any) -> LLMResponse:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        kwargs.pop("max_tokens", None)   # Gemini uses max_output_tokens instead

        self.log.debug("→ Gemini  model=%s  prompt_len=%d", self.model_name, len(full_prompt))
        resp = self.model.generate_content(full_prompt, **kwargs)
        text = resp.text

        self.log.debug("← Gemini  reply_len=%d", len(text))
        return LLMResponse(text=text)
