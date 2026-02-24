"""Prompt templates for mental health classification evaluation."""


# ── V1: Original zero-shot prompt (deprecated) ──────────────────────────────
# Kept for reference and A/B comparison.

CLASSIFICATION_PROMPT_V1 = """You are evaluating a social media post for signs of depression.

Read the following post carefully and classify whether the author is expressing signs of depression or not.

Respond with ONLY one of these two labels, nothing else:
- Depressed
- Not Depressed

Post:
\"\"\"{post_text}\"\"\"

Classification:"""


# ── V2: Enhanced prompt with clinical framework ─────────────────────────────
# Provides explicit depression indicators, handles edge cases like sarcasm
# and casual usage, and accounts for multilingual / transliterated text.

CLASSIFICATION_PROMPT_V2 = """You are a mental health text classifier analyzing social media posts for signs of depression.

Consider these indicators of depression:
- Expressions of hopelessness, worthlessness, or emptiness
- Social withdrawal or isolation ("no one cares", "always alone")
- Loss of interest or pleasure in activities
- Sleep disturbances, fatigue, or low energy
- Negative self-talk or self-harm references
- Persistent sadness beyond normal situational reactions

Do NOT classify as depressed:
- Sarcasm or dark humor without genuine distress
- Temporary frustration or complaining about a specific event
- Song lyrics, quotes, or reposted content
- Using the word "depressed" casually ("this weather is depressing")

The post may be in any language or script (including transliterated text).

Classify the following post. Respond with ONLY one label:
- Depressed
- Not Depressed

Post:
\"\"\"{post_text}\"\"\"

Classification:"""


# ── V3: Few-shot prompt with examples ───────────────────────────────────────
# Based on error analysis of Urdu classification results. Includes 6 labeled
# examples targeting the most common failure modes: behavioral/somatic
# depression, political commentary FPs, poetry/shayri FPs, and short posts.
# Examples sourced from the full dataset but NOT present in the 500-sample
# test set (verified) to avoid data leakage.

CLASSIFICATION_PROMPT_V3 = """You are a mental health text classifier analyzing social media posts for signs of depression.

Consider these indicators of depression:
- Expressions of hopelessness, worthlessness, or emptiness
- Social withdrawal or isolation ("no one cares", "always alone")
- Loss of interest or pleasure in activities
- Sleep disturbances, fatigue, or low energy
- Negative self-talk or self-harm references
- Persistent sadness beyond normal situational reactions
- Behavioral symptoms: lethargy, neglecting self-care, appetite/sleep changes — even without explicitly sad language

Do NOT classify as depressed:
- Sarcasm or dark humor without genuine distress
- Temporary frustration or complaining about a specific event
- Song lyrics, quotes, poetry, or reposted content
- Using emotional words casually ("this weather is depressing")
- Political/societal commentary that uses intense emotional language but is not self-directed
- Third-person references to someone else's suffering

The post may be in any language or script (including Roman Urdu, transliterated text, or code-switched text mixing languages).

Here are examples to guide your classification:

Example 1:
Post: "aj mujhe bahut susti mehsos rahi kuch krne bilkul mod nahy"
Classification: Depressed
(Describes lethargy and complete lack of motivation — behavioral indicators of depression even without explicitly sad language.)

Example 2:
Post: "lakh g lanat hoggahi bilawal pehingi siasat j lahi wrri sindh khe mqm khe wikro krahi dio paya lanat hojahi tohinjahi taqder te begherat insan"
Classification: Not Depressed
(Political commentary expressing frustration about politicians. Uses strong emotional language but is about societal issues, not personal mental health.)

Example 3:
Post: "mera hr din teri hr rat acha hoga meri hr bat teri shayri achi hohi dekh lena gamgen ankho ek din mera janaza teri brat acha hoga"
Classification: Not Depressed
(Ghazal/shayri-style romantic poetry. Despite mentioning "gamgen ankho" (sorrowful eyes) and "janaza" (funeral), this is literary expression using conventional poetic vocabulary, not personal distress.)

Example 4:
Post: "hr pal hr sans mayosii andheron mahinn doba hahi mera dil"
Classification: Depressed
(Every moment, every breath, the heart drowning in despair and darkness. Although poetic in register, this expresses persistent personal hopelessness — a genuine expression of suffering, not a literary exercise.)

Example 5:
Post: "koyi hahi mere sath"
Classification: Not Depressed
(A short, ambiguous post meaning "is anyone with me" — this is a casual social query, not an expression of isolation or distress.)

Example 6:
Post: "mayosion saya meri zindagi hahi hr kadam thokr khata hon"
Classification: Depressed
(The shadow of hopelessness over my life, stumbling at every step. Expresses persistent despair and helplessness about one's own life.)

Now classify the following post. Respond with ONLY one label:
- Depressed
- Not Depressed

Post:
\"\"\"{post_text}\"\"\"

Classification:"""


# ── Prompt registry ─────────────────────────────────────────────────────────

PROMPTS = {
    "v1": CLASSIFICATION_PROMPT_V1,
    "v2": CLASSIFICATION_PROMPT_V2,
    "v3": CLASSIFICATION_PROMPT_V3,
}

# ── Active prompt ────────────────────────────────────────────────────────────
# Change this to switch which prompt the pipeline uses.

CLASSIFICATION_PROMPT = CLASSIFICATION_PROMPT_V2


def build_prompt(post_text: str, prompt_version: str = None) -> str:
    """Build the classification prompt for a given post."""
    template = PROMPTS.get(prompt_version, CLASSIFICATION_PROMPT) if prompt_version else CLASSIFICATION_PROMPT
    return template.format(post_text=post_text)
