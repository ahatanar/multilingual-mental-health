"""OpenAI / ChatGPT model provider."""

from openai import OpenAI
from .base import ModelProvider


class OpenAIProvider(ModelProvider):
    """
    LLM provider for OpenAI ChatGPT models.
    
    Default model: gpt-4o-mini
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", **kwargs):
        super().__init__(api_key=api_key, model_name=model_name, **kwargs)
        self.client = OpenAI(api_key=self.api_key)

    def _call_api(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a mental health text classifier. Respond concisely."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
