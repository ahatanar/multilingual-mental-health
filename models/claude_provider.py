"""Anthropic Claude model provider."""

import anthropic
from .base import ModelProvider


class ClaudeProvider(ModelProvider):
    """
    LLM provider for Anthropic Claude models.

    Default model: claude-3-5-haiku-20241022
    """

    def __init__(self, api_key: str, model_name: str = "claude-haiku-4-5", **kwargs):
        super().__init__(api_key=api_key, model_name=model_name, **kwargs)
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def _call_api(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=50,
            messages=[
                {"role": "user", "content": prompt},
            ],
            system="You are a mental health text classifier. Respond concisely.",
            temperature=0.0,
        )
        return response.content[0].text
