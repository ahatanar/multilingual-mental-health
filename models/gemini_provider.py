"""Google Gemini model provider."""

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from .base import ModelProvider

SAFETY_OFF = {
    HarmCategory.HARM_CATEGORY_HARASSMENT:        HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH:       HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


class GeminiProvider(ModelProvider):
    """
    LLM provider for Google Gemini models.

    Uses the google-generativeai SDK.
    Default model: gemini-3.1-flash-lite
    Safety filters are disabled so the model responds to all mental health content.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-3.1-flash-lite", **kwargs):
        super().__init__(api_key=api_key, model_name=model_name, **kwargs)
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def _call_api(self, prompt: str) -> str:
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=self.max_tokens,
            ),
            safety_settings=SAFETY_OFF,
        )
        return response.text
