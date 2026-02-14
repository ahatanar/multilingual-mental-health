"""Google Gemini model provider."""

import google.generativeai as genai
from .base import ModelProvider


class GeminiProvider(ModelProvider):
    """
    LLM provider for Google Gemini models.
    
    Uses the google-generativeai SDK.
    Default model: gemini-2.0-flash
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash", **kwargs):
        super().__init__(api_key=api_key, model_name=model_name, **kwargs)
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def _call_api(self, prompt: str) -> str:
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=50,
            ),
        )
        return response.text
