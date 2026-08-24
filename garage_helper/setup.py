"""
Interactive LLM setup for notebooks.

Two modes:
  Contributors / repo owners — keep a .env at the repo root with provider config.
    setup_llm() loads it silently and returns the configured model name.

  Learners with no config — setup_llm() runs an interactive wizard that asks
    for a provider, credentials, and model name, sets os.environ for the current
    session, and returns the model name ready to pass into LLMRouter.

Usage (notebook setup cell):
    from garage_helper import setup_llm, LLMRouter
    MODEL  = setup_llm()
    router = LLMRouter(default_model=MODEL)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Provider catalogue
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, dict] = {
    "1": {
        "label": "Anthropic (Claude direct API)",
        "env": {
            "ANTHROPIC_API_KEY": {"prompt": "Anthropic API key", "secret": True},
        },
        "default_model": "claude-haiku-4-5",
        "model_hint": "e.g. claude-haiku-4-5 / claude-sonnet-5 / claude-opus-4-8",
    },
    "2": {
        "label": "OpenAI (direct API)",
        "env": {
            "OPENAI_API_KEY":  {"prompt": "OpenAI API key", "secret": True},
            "OPENAI_PROVIDER": {"prompt": None, "secret": False, "fixed": "openai"},
        },
        "default_model": "gpt-4o-mini",
        "model_hint": "e.g. gpt-4o-mini / gpt-4o / gpt-3.5-turbo",
    },
    "3": {
        "label": "Azure OpenAI",
        "env": {
            "AZURE_OPENAI_ENDPOINT": {"prompt": "Azure OpenAI endpoint URL", "secret": False},
            "AZURE_OPENAI_API_KEY":  {"prompt": "Azure OpenAI API key", "secret": True},
            "OPENAI_PROVIDER":       {"prompt": None, "secret": False, "fixed": "azure"},
        },
        "default_model": "gpt-4o",
        "model_hint": "Azure deployment name, e.g. gpt-4o / gpt-35-turbo",
    },
    "4": {
        "label": "AWS Bedrock (Claude via Bedrock)",
        "env": {
            "AWS_ACCESS_KEY_ID":     {"prompt": "AWS Access Key ID", "secret": True},
            "AWS_SECRET_ACCESS_KEY": {"prompt": "AWS Secret Access Key", "secret": True},
            "AWS_DEFAULT_REGION":    {
                "prompt": "AWS region",
                "secret": False,
                "default": "us-east-1",
            },
        },
        "default_model": "anthropic.claude-3-5-haiku-20241022-v1:0",
        "model_hint": "e.g. anthropic.claude-3-5-haiku-20241022-v1:0",
    },
    "5": {
        "label": "Google Gemini",
        "env": {
            "GOOGLE_API_KEY": {"prompt": "Google API key", "secret": True},
        },
        "default_model": "gemini-2.0-flash",
        "model_hint": "e.g. gemini-2.0-flash / gemini-1.5-pro",
    },
    "6": {
        "label": "DeepSeek",
        "env": {
            "DEEPSEEK_API_KEY": {"prompt": "DeepSeek API key", "secret": True},
        },
        "default_model": "deepseek-chat",
        "model_hint": "e.g. deepseek-chat / deepseek-reasoner",
    },
    "7": {
        "label": "Ollama (local)",
        "env": {
            "OLLAMA_BASE_URL": {
                "prompt": "Ollama base URL",
                "secret": False,
                "default": "http://localhost:11434/v1",
            },
        },
        "default_model": "ollama/llama3.2",
        "model_hint": "e.g. ollama/llama3.2 / ollama/mistral",
    },
}

_ENV_FILE = ".env"
_DEFAULT_MODEL_ENV = "DEFAULT_LLM_MODEL"


def _load_dotenv(path: Path) -> bool:
    """Parse a .env file and inject values into os.environ. Returns True if the file existed."""
    if not path.is_file():
        return False
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:  # don't overwrite existing shell env
            os.environ[key] = val
    return True


def _save_dotenv(path: Path, keys: list) -> None:
    """Merge the given env var keys into the .env file, creating it if needed."""
    existing: dict = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            existing[k.strip()] = v.strip()

    for key in keys:
        val = os.environ.get(key, "")
        if val:
            existing[key] = val

    path.write_text("\n".join(f"{k}={v}" for k, v in existing.items()) + "\n")
    print(f"  Credentials saved to {path} — won't be asked again.")


def _find_repo_root() -> Path:
    """Walk up from cwd looking for pyproject.toml or a .env file."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".env").exists():
            return parent
    return cwd


def _in_notebook() -> bool:
    """Return True when running inside a Jupyter/VSCode notebook kernel."""
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"  # type: ignore[name-defined]
    except NameError:
        return False


def _read_secret(prompt_text: str) -> str:
    """Read a secret value.

    Notebooks (VSCode/Colab/JupyterLab): uses input() — value appears in the
    input box at the top of the window and is not echoed to cell output.
    Terminal: uses getpass for hidden input.
    """
    if _in_notebook():
        return input(f"  {prompt_text}: ").strip()
    try:
        import getpass
        return getpass.getpass(f"  {prompt_text}: ").strip()
    except Exception:
        return input(f"  {prompt_text}: ").strip()


def _wizard() -> tuple:
    """Run the interactive provider wizard. Returns (model_name, env_keys_to_save)."""
    print("\n── LLM Setup Wizard ──────────────────────────────────────")
    print("No LLM config found. Let's get you set up.\n")
    print("Choose your LLM provider:")
    for key, p in _PROVIDERS.items():
        print(f"  {key}. {p['label']}")
    print()

    choice = input("Enter number (1–7): ").strip()
    while choice not in _PROVIDERS:
        choice = input("Please enter a number between 1 and 7: ").strip()

    provider = _PROVIDERS[choice]
    print(f"\nSelected: {provider['label']}\n")

    keys_to_save = [_DEFAULT_MODEL_ENV]
    for env_key, spec in provider["env"].items():
        if "fixed" in spec:
            os.environ[env_key] = spec["fixed"]
            keys_to_save.append(env_key)
            continue

        if os.environ.get(env_key, "").strip():
            print(f"  {env_key} already set — keeping existing value.")
            keys_to_save.append(env_key)
            continue

        default = spec.get("default", "")
        hint    = f" [{default}]" if default else ""
        label   = spec["prompt"] + hint

        value = _read_secret(label) if spec.get("secret") else input(f"  {label}: ").strip()
        if not value and default:
            value = default
        if value:
            os.environ[env_key] = value
            keys_to_save.append(env_key)

    print(f"\n  {provider['model_hint']}")
    model_input = input(f"  Model name [leave blank for {provider['default_model']}]: ").strip()
    model = model_input or provider["default_model"]

    os.environ[_DEFAULT_MODEL_ENV] = model
    print(f"\n  Model set to: {model}")
    return model, keys_to_save


def _smoke_test(model: str) -> bool:
    """Make one cheap API call to confirm credentials work."""
    from .router import LLMRouter
    print(f"  Verifying credentials with a test call... ", end="", flush=True)
    try:
        reply = LLMRouter(default_model=model).generate(
            "Reply with the single word: ready", model=model, max_tokens=5
        )
        ok = "ready" in reply.lower()
        print("OK" if ok else f"unexpected reply: {reply!r}")
        return ok
    except Exception as exc:
        print(f"FAILED\n  {exc}")
        return False


def setup_llm(
    env_file: Optional[str] = None,
    smoke_test: bool = True,
    verbose: bool = False,
) -> str:
    """
    Set up LLM credentials and return the model name to use.

    1. Loads a .env file from the repo root (or env_file if given).
    2. If DEFAULT_LLM_MODEL is already in the environment, returns it.
    3. Otherwise runs an interactive wizard to collect provider + credentials.
    4. Smoke-tests the chosen model before returning (disable with smoke_test=False).

    Parameters
    ----------
    env_file : str, optional
        Explicit path to a .env file. Defaults to repo-root .env.
    smoke_test : bool
        Make one test call to confirm credentials work (default True).
    verbose : bool
        Print extra debug lines.

    Returns
    -------
    str
        Model name ready to pass as ``LLMRouter(default_model=MODEL)``.
    """
    env_path = Path(env_file) if env_file else _find_repo_root() / _ENV_FILE
    loaded   = _load_dotenv(env_path)
    if loaded and verbose:
        print(f"Loaded config from {env_path}")

    model = os.environ.get(_DEFAULT_MODEL_ENV, "").strip()
    if model:
        if verbose:
            print(f"Using model from environment: {model}")
        if smoke_test:
            _smoke_test(model)
        return model

    model, keys_to_save = _wizard()

    # Save immediately so the wizard is never repeated, even if the smoke test fails.
    _save_dotenv(env_path, keys_to_save)

    if smoke_test:
        ok = _smoke_test(model)
        if not ok:
            print(
                "\n  The test call failed. Check your credentials and try again.\n"
                "  Edit the .env file at the repo root to update them, then re-run this cell."
            )

    print(f"\n── Setup complete. Using: {model} ─────────────────────────\n")
    return model
