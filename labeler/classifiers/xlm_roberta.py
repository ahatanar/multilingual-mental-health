"""XLM-RoBERTa depression classifier (local HuggingFace model)."""

import logging

from transformers import pipeline, AutoTokenizer

from .base import LabelerClassifier

logger = logging.getLogger(__name__)

MODEL_ID = "malexandersalazar/xlm-roberta-base-cls-depression"
BASE_TOKENIZER = "xlm-roberta-base"


class XLMRobertaClassifier(LabelerClassifier):
    """
    Local depression classifier using XLM-RoBERTa fine-tuned for depression.

    Supports Spanish natively (no translation needed).
    For Chinese, expects English-translated text.

    Model outputs:
        LABEL_1 (depression) -> "depressed"
        LABEL_0 (no depression) -> "not depressed"
    """

    def __init__(self):
        logger.info(f"Loading XLM-RoBERTa model: {MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
        self._pipe = pipeline("text-classification", model=MODEL_ID, tokenizer=tokenizer)
        logger.info("XLM-RoBERTa model loaded successfully")

    def classify(self, text: str) -> str:
        result = self._pipe(text, truncation=True, max_length=512)[0]
        label = result["label"]
        # LABEL_1 = depression, LABEL_0 = no depression
        if label == "LABEL_1":
            return "depressed"
        return "not depressed"
