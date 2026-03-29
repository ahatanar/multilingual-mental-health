"""Configuration — reads API keys from environment variables."""

import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

# ── Local model keys ──────────────────────────────────────────────────────────
# Models in this set use LM Studio and need no real API key.
# If you add a new local model entry to runner.py MODELS, add its key here too.
LOCAL_PROVIDERS = {"lmstudio", "llama", "qwen", "mistral"}

# ── Online API key mapping ────────────────────────────────────────────────────
# Maps runner.py MODELS keys -> environment variable names.
# Add a new entry here when you add a new online model.
ENV_MAP = {
    "gemini":   "GEMINI_API_KEY",
    "openai":   "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "claude":   "CLAUDE_API_KEY",
    "grok":     "GROK_API_KEY",
    "cohere":   "COHERE_API_KEY",
}


def get_api_key(provider: str) -> str:
    """
    Return the API key for a given provider.

    Local providers (LM Studio) return a dummy key — no env var needed.
    Online providers read from environment variables (loaded from .env).
    """
    if provider.lower() in LOCAL_PROVIDERS:
        return "lm-studio"

    env_var = ENV_MAP.get(provider.lower())
    if not env_var:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            f"Add it to ENV_MAP in config.py or LOCAL_PROVIDERS if it's a local model."
        )

    key = os.environ.get(env_var)
    if not key:
        raise EnvironmentError(
            f"API key not found. Set {env_var} in your .env file.\n"
            f"  {env_var}=your-key-here"
        )
    return key


# Default model names per provider (used outside of runner.py)
DEFAULT_MODELS = {
    "gemini":   "gemini-2.0-flash",
    "openai":   "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "claude":   "claude-haiku-4-5-20241022",
}