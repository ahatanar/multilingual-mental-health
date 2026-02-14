"""DeepSeek model provider using OpenAI-compatible API."""

from openai import OpenAI
from .base import ModelProvider


class DeepSeekProvider(ModelProvider):
    """
    LLM provider for DeepSeek models.
    
    Uses OpenAI-compatible API at https://api.deepseek.com
    Default model: deepseek-chat
    """

    def __init__(self, api_key: str, model_name: str = "deepseek-chat", **kwargs):
        super().__init__(api_key=api_key, model_name=model_name, **kwargs)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
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
