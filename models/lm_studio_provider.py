"""LM Studio local model provider using OpenAI-compatible API."""

from openai import OpenAI
from .base import ModelProvider


class LMStudioProvider(ModelProvider):
    """
    LLM provider for models running locally via LM Studio.

    LM Studio exposes an OpenAI-compatible server at http://localhost:1234/v1.
    No real API key is required — pass any string (e.g. "lm-studio").

    The model_name must match the model identifier shown in LM Studio's
    "Local Server" tab (e.g. "llama-3.2-3b-instruct", "mistral-7b-instruct-v0.3").
    """

    def __init__(self, api_key: str = "lm-studio", model_name: str = "local-model",
                 base_url: str = "http://localhost:1234/v1", **kwargs):
        super().__init__(api_key=api_key, model_name=model_name, **kwargs)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )

    def _call_api(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a mental health text classifier. Respond concisely."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        return response.choices[0].message.content
