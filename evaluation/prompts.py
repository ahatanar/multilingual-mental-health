"""Prompt templates for mental health classification evaluation."""


# Zero-shot classification prompt — English, consistent across all models.
# The model must reply with ONLY "Depressed" or "Not Depressed".
CLASSIFICATION_PROMPT = """You are evaluating a social media post for signs of depression.

Read the following post carefully and classify whether the author is expressing signs of depression or not.

Respond with ONLY one of these two labels, nothing else:
- Depressed
- Not Depressed

Post:
\"\"\"{post_text}\"\"\"

Classification:"""


def build_prompt(post_text: str) -> str:
    """Build the classification prompt for a given post."""
    return CLASSIFICATION_PROMPT.format(post_text=post_text)
