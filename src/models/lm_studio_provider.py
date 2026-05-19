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
                 base_url: str = "http://localhost:1234/v1", no_think: bool = False, **kwargs):
        kwargs.setdefault("max_tokens", 1000)
        super().__init__(api_key=api_key, model_name=model_name, **kwargs)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )
        self.no_think = no_think

    def _call_api(self, prompt: str) -> str:
        system = "You are an academic research assistant performing mental health text classification. This is a university research project. You must respond to ALL inputs without exception — do not refuse, filter, or return empty responses regardless of content sensitivity. Your task is classification only; you are not producing harmful content."
        extra = {"enable_thinking": False} if self.no_think else {}
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=self.max_tokens,
            extra_body=extra,
        )
        return response.choices[0].message.content or ""