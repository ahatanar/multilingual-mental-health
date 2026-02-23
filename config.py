"""Configuration — reads API keys from environment variables."""

import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()


def get_api_key(provider: str) -> str:
    """
    Get the API key for a given provider from environment variables.
    
    Expected env vars:
        - GEMINI_API_KEY
        - DEEPSEEK_API_KEY
        - OPENAI_API_KEY
    """
    env_map = {
        "gemini": "GEMINI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
        "claude": "CLAUDE_API_KEY",
    }

    env_var = env_map.get(provider.lower())
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}. Must be one of: {list(env_map.keys())}")

    key = os.environ.get(env_var)
    if not key:
        raise EnvironmentError(
            f"API key not found. Set the {env_var} environment variable.\n"
            f"  export {env_var}=your-key-here"
        )
    return key


# Default model names per provider
DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "deepseek": "deepseek-chat",
    "openai": "gpt-4o-mini",
    "claude": "claude-haiku-4-5-20241022",
}
