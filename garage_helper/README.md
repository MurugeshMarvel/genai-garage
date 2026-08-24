# garage_helper — The LLM Router

Every experiment in this repo calls an LLM. The problem is that the "right" LLM depends on who you are, where your API keys live, and what you are trying to do. You might be running on Azure OpenAI at work, calling Anthropic's API at home, testing with a local Ollama setup, or benchmarking DeepSeek against GPT.

Without a router, every notebook hard-codes a provider. Change jobs, change keys, move to a different cloud — and you are hunting through a dozen notebooks updating import paths and client initialisation code. That is not engineering. That is maintenance debt.

The router solves this once. You write one line. The router figures out the rest.

---

## Why a Router?

Three reasons.

**First: provider diversity.** This repo supports seven providers across five families — Azure OpenAI, direct OpenAI, Claude via AWS Bedrock, Claude via Anthropic's API, Google Gemini, Ollama (local), and DeepSeek. Each has a different SDK, different client initialisation, different request format, and different response shape. The router hides all of that behind a single `generate()` call.

**Second: zero-config switching.** Most notebooks just want *a response from a model*. They should not care which cloud it came from. The router reads your environment variables and routes automatically. Set `OPENAI_ENDPOINT` and you get Azure. Set `OPENAI_PROVIDER=openai` and you get direct OpenAI. Pass a Bedrock model ID and you get Bedrock. No code changes required.

**Third: observable by default.** When something breaks — and it will — you want to know exactly what prompt was sent, which model received it, and how many tokens came back. Setting `verbose=True` turns on structured logging across every provider with a single flag. It's off by default so notebooks stay clean.

---

## How It Works

```
Your code
    │
    ▼
LLMRouter.generate(prompt, model="gpt-4o")
    │
    ├── 1. Resolve model → provider key
    │       "gpt-4o"                          → auto_openai
    │       "anthropic.claude-sonnet-..."      → bedrock_claude
    │       "claude-sonnet-5"                 → anthropic
    │       "gemini-2.0-flash"                → gemini
    │       "deepseek-chat"                   → deepseek
    │       "ollama/llama3.2"                 → ollama
    │
    ├── 2. auto_openai: pick Azure vs direct OpenAI
    │       OPENAI_PROVIDER=azure?            → AzureOpenAIProvider
    │       OPENAI_PROVIDER=openai?           → OpenAIProvider
    │       (neither) OPENAI_ENDPOINT set?    → AzureOpenAIProvider
    │       (nothing set)                     → OpenAIProvider
    │
    ├── 3. Retrieve (or create) a cached provider instance
    │       Providers are instantiated once per (model, provider) pair.
    │       Repeated calls to generate() reuse the same client object.
    │
    └── 4. Call provider.generate(prompt, system=..., **kwargs)
                Returns an LLMResponse(text, input_tokens, output_tokens, stop_reason)
```

There are three files doing the work:

| File | Responsibility |
|---|---|
| `router.py` | Model registry, provider resolution, the `LLMRouter` class |
| `providers.py` | One class per provider — SDK calls, error handling, debug logging |
| `logging_config.py` | Shared logger; `verbose=True` sets it to DEBUG, False keeps it at WARNING |

---

## Supported Providers

| Provider key | When it's chosen | Required env vars |
|---|---|---|
| `azure_openai` | GPT model + `OPENAI_PROVIDER=azure` **or** `OPENAI_ENDPOINT` set | `OPENAI_ENDPOINT`, `OPENAI_API_KEY` |
| `openai` | GPT model + `OPENAI_PROVIDER=openai` **or** no endpoint set | `OPENAI_API_KEY` |
| `bedrock_claude` | Model ID is a Bedrock ARN or starts with `anthropic.claude-` | AWS credentials via `~/.aws` or `AWS_*` env vars |
| `anthropic` | Model ID starts with `claude-` (no `anthropic.` prefix) | `ANTHROPIC_API_KEY` |
| `gemini` | Model ID starts with `gemini-` | `GOOGLE_API_KEY` |
| `deepseek` | Model ID is `deepseek-chat` or `deepseek-reasoner` | `DEEPSEEK_API_KEY` |
| `ollama` | Model ID starts with `ollama/` | None (local server at `OLLAMA_BASE_URL`, default `http://localhost:11434/v1`) |

The router never reads credentials directly — it reads standard environment variables and delegates to each SDK. Keep your keys in `.env` and load them with `python-dotenv` or just export them in your shell.

---

## Usage

### The minimum you need to know

```python
from garage_helper import LLMRouter

router = LLMRouter()
reply = router.generate("Explain attention in one sentence.")
print(reply)
```

That's it. The router picks the default model (Bedrock Claude unless you override it), resolves the provider from your environment, and returns a string.

---

### Choosing a model per call

```python
router = LLMRouter()

# GPT via Azure OpenAI (needs OPENAI_ENDPOINT + OPENAI_API_KEY)
reply = router.generate("Summarise this.", model="gpt-4o")

# GPT via direct OpenAI (needs OPENAI_API_KEY)
os.environ["OPENAI_PROVIDER"] = "openai"
reply = router.generate("Summarise this.", model="gpt-4o")

# Claude via Bedrock
reply = router.generate("Summarise this.",
                        model="anthropic.claude-3-5-sonnet-20241022-v2:0")

# Claude via Anthropic direct API
reply = router.generate("Summarise this.", model="claude-sonnet-5")

# Gemini
reply = router.generate("Summarise this.", model="gemini-2.0-flash")

# DeepSeek (needs DEEPSEEK_API_KEY)
reply = router.generate("Summarise this.", model="deepseek-chat")

# Ollama (local, no API key)
reply = router.generate("Summarise this.", model="ollama/llama3.2")
```

---

### Switching between Azure and direct OpenAI for GPT models

GPT model names (`gpt-4o`, `gpt-4o-mini`, etc.) are deliberately provider-agnostic — the same name works on both Azure and direct OpenAI. The router picks the right backend based on this priority:

```
OPENAI_PROVIDER=azure   →  always Azure OpenAI  (needs OPENAI_ENDPOINT)
OPENAI_PROVIDER=openai  →  always direct OpenAI  (needs OPENAI_API_KEY)
(neither set)           →  Azure if OPENAI_ENDPOINT exists, else direct OpenAI
```

Switch once per environment; notebooks stay unchanged.

```bash
# Option A: force Azure
export OPENAI_PROVIDER=azure
export OPENAI_ENDPOINT=https://my-resource.openai.azure.com
export OPENAI_API_KEY=...

# Option B: force direct OpenAI
export OPENAI_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

```python
router = LLMRouter()
reply = router.generate("Hello.", model="gpt-4o")  # same code, different backend
```

---

### Setting a default model for a notebook

If every cell in a notebook uses the same model, set it once at the top:

```python
from garage_helper import LLMRouter

router = LLMRouter(default_model="anthropic.claude-3-5-sonnet-20241022-v2:0")

# Every generate() call below uses Bedrock Claude unless overridden
reply = router.generate("What is RAG?")
```

Or use the environment variable so the notebook has no hard-coded model at all:

```bash
export DEFAULT_LLM_MODEL="gemini-2.0-flash"
```

```python
router = LLMRouter()   # picks up DEFAULT_LLM_MODEL from the environment
```

---

### System prompts

```python
router = LLMRouter()
reply = router.generate(
    prompt="List three reasons why RAG beats fine-tuning for knowledge-intensive tasks.",
    system="You are a concise technical writer. Use bullet points.",
)
```

---

### Verbose logging

Turn on `verbose=True` to see every request and response in your terminal or Jupyter cell. Useful when debugging prompts or checking token counts.

```python
router = LLMRouter(verbose=True)
reply = router.generate("What is layer normalisation?")
```

Output looks like:

```
[14:32:01] DEBUG    — LLMRouter ready  default_model=anthropic.claude-3-5-haiku-20241022-v1:0
[14:32:01] DEBUG    — auto_openai → azure_openai  (OPENAI_PROVIDER='(unset)')
[14:32:01] DEBUG    — generate()  model=gpt-4o  provider=azure_openai  prompt_len=31
[14:32:01] DEBUG    — → AzureOpenAI  model=gpt-4o  prompt_len=31
[14:32:02] DEBUG    — ← AzureOpenAI  tokens=142  reply_len=618
```

Set `verbose=False` (the default) and nothing is printed.

---

### Passing kwargs to the SDK

Any extra keyword argument you pass to `generate()` is forwarded to the underlying SDK call. This covers `temperature`, `max_tokens`, `top_p`, and anything else the provider supports.

```python
reply = router.generate(
    "Write a creative product name for an AI scheduling tool.",
    model="gpt-4o",
    temperature=0.9,
    max_tokens=50,
)
```

---

### Getting token counts back

```python
from garage_helper import LLMRouter

router = LLMRouter()
resp = router.generate_response("Explain RAG in one paragraph.", max_tokens=200)
print(resp.text)
print(f"Tokens used: {resp.input_tokens} in / {resp.output_tokens} out / {resp.total_tokens} total")
print(f"Stop reason: {resp.stop_reason}")
```

---

### Introspection

```python
router = LLMRouter()

# See all registered model IDs
print(router.list_models())

# Check which provider a model name routes to
print(router.provider_for("gemini-2.0-flash"))     # → "gemini"
print(router.provider_for("gpt-4o"))                # → "azure_openai" or "openai" (depends on env)
print(router.provider_for("deepseek-chat"))         # → "deepseek"
print(router.provider_for("ollama/mistral"))        # → "ollama"
```

---

## Environment Variables Reference

| Variable | Purpose |
|---|---|
| `OPENAI_PROVIDER` | Explicit GPT backend: `azure` or `openai`. Overrides endpoint detection. |
| `OPENAI_ENDPOINT` | Azure OpenAI endpoint URL. Auto-selects Azure when set (and `OPENAI_PROVIDER` is not). |
| `OPENAI_API_KEY` | API key for Azure OpenAI or direct OpenAI. |
| `AZURE_OPENAI_ENDPOINT` | Alternative name for `OPENAI_ENDPOINT`. |
| `AZURE_OPENAI_API_KEY` | Alternative name for `OPENAI_API_KEY`. |
| `ANTHROPIC_API_KEY` | Anthropic direct API key. |
| `GOOGLE_API_KEY` | Google AI / Gemini API key. |
| `DEEPSEEK_API_KEY` | DeepSeek API key. |
| `OLLAMA_BASE_URL` | Ollama server URL (default: `http://localhost:11434/v1`). |
| `AWS_DEFAULT_REGION` | AWS region for Bedrock (defaults to `us-east-1`). |
| `AWS_ACCESS_KEY_ID` | AWS access key (if not using `~/.aws/credentials`). |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key (if not using `~/.aws/credentials`). |
| `DEFAULT_LLM_MODEL` | Default model ID when none is specified in code. |
| `BEDROCK_INFERENCE_PROFILE_ARN` | Your account's Bedrock application inference profile ARN, if you route through one instead of a plain model ID. Account-specific — keep it in `.env`, never in code or notebooks. |

---

## Adding a New Provider

The fastest path is to subclass `OpenAICompatibleProvider` — it handles the OpenAI client wiring, debug logging, and response normalisation. You just supply a base URL and the env var that holds the API key:

```python
# In providers.py — add this class (5 lines)
class GroqProvider(OpenAICompatibleProvider):
    name = "groq"
    def __init__(self, model: str, verbose: bool = False):
        super().__init__(model, base_url="https://api.groq.com/openai/v1",
                         api_key_env="GROQ_API_KEY", verbose=verbose)
```

Then wire it into the router (2 lines in `router.py`):

```python
# In _PROVIDER_MAP
"groq": GroqProvider,

# In _MODEL_REGISTRY
"llama-3.3-70b-versatile": "groq",
"mixtral-8x7b-32768":      "groq",
```

That's the full surface area. Nothing in the notebooks needs to change.

Providers that *don't* speak the OpenAI API (e.g. a custom REST endpoint) should instead subclass nothing and implement `generate(self, prompt, system=None, **kwargs) -> LLMResponse` directly, following the pattern of `BedrockClaudeProvider`.

---

## Adding a New Model

If the model belongs to an existing provider, add one line to `_MODEL_REGISTRY` in `router.py`:

```python
"gpt-4o-2024-11-20": "auto_openai",
```

Models not in the registry fall through to best-effort guessing:
- ARN starting with `arn:aws:bedrock` → `bedrock_claude`
- Contains `anthropic.` or starts with `us.` + contains `claude` → `bedrock_claude`
- Contains `claude` → `anthropic`
- Contains `gemini` → `gemini`
- Starts with `ollama/` → `ollama`
- Contains `deepseek` → `deepseek`
- Anything else → `auto_openai`
