"""Abstract base class for all LLM model providers."""

from abc import ABC, abstractmethod
import time
import logging

logger = logging.getLogger(__name__)


class ModelProvider(ABC):
    """
    Abstract base for LLM providers used in mental health evaluation.
    
    Each provider wraps a specific LLM API (Gemini, DeepSeek, OpenAI, etc.)
    and exposes a consistent classify() interface.
    """

    def __init__(self, api_key: str, model_name: str, max_retries: int = 3, retry_delay: float = 2.0):
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @property
    def name(self) -> str:
        """Human-readable identifier for this model provider."""
        return self.model_name

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """
        Send a prompt to the LLM and return the raw text response.
        Subclasses must implement this.
        """
        pass

    def classify(self, post: str, prompt_template: str) -> dict:
        """
        Classify a post using the LLM.
        
        Args:
            post: The social media post text to classify.
            prompt_template: The prompt template with a {post_text} placeholder.
            
        Returns:
            dict with keys:
                - prediction: "depressed" or "not depressed"
                - raw_response: the full model response string
                - error: None or error message if something went wrong
        """
        prompt = prompt_template.format(post_text=post)

        for attempt in range(1, self.max_retries + 1):
            try:
                raw_response = self._call_api(prompt)
                prediction = self._parse_response(raw_response)
                return {
                    "prediction": prediction,
                    "raw_response": raw_response,
                    "error": None,
                }
            except Exception as e:
                logger.warning(f"[{self.name}] Attempt {attempt}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)  # exponential-ish backoff
                else:
                    logger.error(f"[{self.name}] All {self.max_retries} attempts failed.")
                    return {
                        "prediction": "error",
                        "raw_response": "",
                        "error": str(e),
                    }

    @staticmethod
    def _parse_response(raw_response: str) -> str:
        """
        Parse the LLM response into a normalized label.
        
        Looks for 'depressed' or 'not depressed' in the response.
        Returns 'depressed', 'not depressed', or 'unclear'.
        """
        text = raw_response.strip().lower()

        # Check for "not depressed" first (more specific)
        not_depressed_signals = ["not depressed", "not_depressed", "no depression", "normal", "non-depressed"]
        for signal in not_depressed_signals:
            if signal in text:
                return "not depressed"

        # Then check for "depressed"
        depressed_signals = ["depressed", "depression", "yes"]
        for signal in depressed_signals:
            if signal in text:
                return "depressed"

        return "unclear"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
