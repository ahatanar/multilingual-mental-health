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


# ── V3 Urdu — Experiment 2: Attribution (explain an existing classification) ──
# The classification has already been made in Experiment 1.
# This prompt asks the model to attribute the decision — to identify the
# specific words that support the given label.
#
# Two placeholders:
#   {prediction}  — the model's own Exp 1 label ("Depressed" / "Not Depressed")
#   {post_text}   — the post text (filled by provider.classify())
#
# Output is exactly TWO lines (no classification line — it is given as input):
#   Line 1: key word(s) in Roman Urdu, comma-separated
#   Line 2: one-word English translation of each, same order

ATTRIBUTION_PROMPT_V3_EXP2 = """You are analyzing a mental health text classification.

A social media post in Roman Urdu has already been classified as "{prediction}".
Your task is to identify the specific word(s) in the post that support this classification.

Notes on Roman Urdu social media:
- Behavioral words (susti = lethargy, neend = sleep) are valid depression signals even without explicitly sad language
- Political commentary words (siasat, bilawal, lanat) indicate non-personal context
- Poetic vocabulary (shayri, janaza, gamgen) may indicate literary register, not personal distress
- Short ambiguous words alone are weak signals

Here are examples. Each shows the post, its classification, and the EXACT two-line response you must produce:

Example 1:
Post: "aj mujhe bahut susti mehsos rahi kuch krne bilkul mod nahy"
Classification: Depressed
susti, mod
lethargy, motivation

Example 2:
Post: "lakh g lanat hoggahi bilawal pehingi siasat j lahi wrri sindh khe mqm khe wikro krahi dio paya lanat hojahi tohinjahi taqder te begherat insan"
Classification: Not Depressed
lanat, siasat, bilawal
curse, politics, politician

Example 3:
Post: "mera hr din teri hr rat acha hoga meri hr bat teri shayri achi hohi dekh lena gamgen ankho ek din mera janaza teri brat acha hoga"
Classification: Not Depressed
janaza, shayri, gamgen
funeral, poetry, sorrowful

Example 4:
Post: "hr pal hr sans mayosii andheron mahinn doba hahi mera dil"
Classification: Depressed
mayosii, andheron, doba
despair, darkness, drowning

Example 5:
Post: "koyi hahi mere sath"
Classification: Not Depressed
koyi, sath
someone, together

Example 6:
Post: "mayosion saya meri zindagi hahi hr kadam thokr khata hon"
Classification: Depressed
mayosion, saya, thokr
hopelessness, shadow, stumble

Now respond for the following. Produce EXACTLY two lines — nothing else:
Line 1: the word(s) from the post that support the classification "{prediction}", in the original Roman Urdu, comma-separated
Line 2: one-word English translation of each, in the same order, comma-separated

Post:
\"\"\"{post_text}\"\"\"

Response:"""


# ── V3 Arabic — Experiment 2: Attribution (explain an existing classification) ─
# Same approach as ATTRIBUTION_PROMPT_V3_EXP2 but for Arabic/Egyptian dialect.

ATTRIBUTION_PROMPT_V3_ARABIC_EXP2 = """You are analyzing a mental health text classification.

A social media post in Arabic (Egyptian dialect) has already been classified as "{prediction}".
Your task is to identify the specific word(s) in the post that support this classification.

Notes on Arabic social media:
- Religious phrases (الحمدالله، إن شاء الله) are everyday expressions, not depression signals
- Seasonal/hashtag posts (رمضان, عيد) indicate community context, not personal distress
- Behavioral absence words (مبتفاعلش = not interacting, مبقتش = no longer) are strong depression signals
- Grief and isolation words (الحزن، الوحدة، الخوف) are direct depression indicators
- Rhetorical questions to followers are social engagement, not isolation

Here are examples. Each shows the post, its classification, and the EXACT two-line response you must produce:

Example 1:
Post: "فقدت الشغف تجاه مجال عملي بسبب كذا صدمة متتالية خاصة اخر صدمة دي دمرتني نفسيا خلتني كرهت كل حاجه كنت بحب مجالي جدا وبطور نفسي فيه باستمرار حاليا مبقتش اطيق افتح كتاب او اقعد قدام فيديو"
Classification: Depressed
فقدت, دمرتني, مبقتش
lost, destroyed, no longer

Example 2:
Post: "الحمدالله حمدا كثيرا"
Classification: Not Depressed
الحمدالله
gratitude

Example 3:
Post: "#رمضان_كريم يارب شهر الخير اقبل بالافراح لا تحرم اللي يرتجي منك غفران انزل علينا رحمة منك نرتاح"
Classification: Not Depressed
رمضان, الافراح
Ramadan, joy

Example 4:
Post: "أشعر بالحزن الشديد بعد وفاة أمي وأشعر بالوحدة لأني لا أملك أصدقاء وأشعر بالخوف من أبي"
Classification: Depressed
الحزن, الوحدة, الخوف
sadness, loneliness, fear

Example 5:
Post: "ايه الحاجه اللي لو حصلتك دلوقتي تخليك #سعيد"
Classification: Not Depressed
سعيد
happy

Example 6:
Post: "والحقيقة آه طلعت وقتها كنت وحشة، مش وحشة بالمعنى الحرفي، انا بس مبتفاعلش معاهم.. بخرج عشان اقعد اسمع اغاني لوحدي"
Classification: Depressed
مبتفاعلش, لوحدي
withdrawn, alone

Now respond for the following. Produce EXACTLY two lines — nothing else:
Line 1: the word(s) from the post that support the classification "{prediction}", in the original Arabic, comma-separated
Line 2: one-word English translation of each, in the same order, comma-separated

Post:
\"\"\"{post_text}\"\"\"

Response:"""


# ── V3 Arabic: Few-shot prompt with Egyptian Arabic examples ─────────────────
# Mirrors V3's structure but with Arabic-specific failure mode guidance:
#   - Religious expressions (الحمد لله, إن شاء الله) are comfort phrases, not depression
#   - Seasonal/community posts (Ramadan greetings, hashtags) are not personal distress
#   - Egyptian slang intensity does not equal clinical depression
#   - Arabic poetry/literary registers vs genuine personal suffering
# Examples drawn from the filtered dataset (NOT present in the 5000-sample test set).

CLASSIFICATION_PROMPT_V3_ARABIC = """You are a mental health text classifier analyzing social media posts for signs of depression.

Consider these indicators of depression:
- Expressions of hopelessness, worthlessness, or emptiness
- Social withdrawal or isolation ("no one cares", "always alone")
- Loss of interest or pleasure in activities
- Sleep disturbances, fatigue, or low energy
- Negative self-talk or self-harm references
- Persistent sadness beyond normal situational reactions
- Behavioral symptoms: loss of passion, neglecting responsibilities, inability to engage — even without explicitly sad language

Do NOT classify as depressed:
- Religious expressions or phrases (الحمد لله، إن شاء الله، السلام عليكم، يارب) — these are everyday Arabic greetings and expressions of faith, not distress
- Seasonal or community posts (Ramadan greetings, holiday posts, religious hashtags) — celebratory or communal, not personal suffering
- Relationship advice or venting about situational problems — temporary frustration, not clinical depression
- Interactive/rhetorical questions directed at followers — social engagement, not isolation
- Arabic poetry, song lyrics, or romantic language — literary expression, not personal distress
- Sarcasm or dark humor without genuine emotional distress

The post may be in Arabic script, Egyptian dialect (slang), standard Arabic, or a mix.

Here are examples to guide your classification:

Example 1:
Post: "فقدت الشغف تجاه مجال عملي بسبب كذا صدمة متتالية خاصة اخر صدمة دي دمرتني نفسيا خلتني كرهت كل حاجه كنت بحب مجالي جدا وبطور نفسي فيه باستمرار حاليا مبقتش اطيق افتح كتاب او اقعد قدام فيديو"
Classification: Depressed
(Lost all passion for work after repeated psychological shocks — behavioral depression indicator even without the word "sad".)

Example 2:
Post: "الحمدالله حمدا كثيرا"
Classification: Not Depressed
(A short religious phrase meaning "Praise be to God." Extremely common in Arabic social media as a simple expression of gratitude or faith — not a sign of depression.)

Example 3:
Post: "#رمضان_كريم يارب شهر الخير اقبل بالافراح لا تحرم اللي يرتجي منك غفران انزل علينا رحمة منك نرتاح"
Classification: Not Depressed
(Ramadan greeting with religious prayer. Seasonal community post — celebratory and collective, not personal distress despite emotional religious language.)

Example 4:
Post: "أشعر بالحزن الشديد بعد وفاة أمي وأشعر بالوحدة لأني لا أملك أصدقاء وأشعر بالخوف من أبي"
Classification: Depressed
(Intense sadness after bereavement, compounded by total social isolation and fear. Multiple simultaneous depression indicators clearly about the author's own suffering.)

Example 5:
Post: "ايه الحاجه اللي لو حصلتك دلوقتي تخليك #سعيد"
Classification: Not Depressed
(Meaning "What would make you happy right now?" — a casual interactive question to followers with a happiness hashtag. Social engagement post, not personal distress.)

Example 6:
Post: "والحقيقة آه طلعت وقتها كنت وحشة، مش وحشة بالمعنى الحرفي، انا بس مبتفاعلش معاهم.. بخرج عشان اقعد اسمع اغاني لوحدي"
Classification: Depressed
(Describes persistent social withdrawal — not interacting with others, choosing to be alone. Behavioral isolation is a key depression indicator even when described casually.)

Now classify the following post. Respond with ONLY one label:
- Depressed
- Not Depressed

Post:
\"\"\"{post_text}\"\"\"

Classification:"""


# ── V3 Chinese: Placeholder prompt (examples pending dataset) ────────────────
# Framework is ready. Language-specific examples will be added once the
# Chinese dataset (data/raw/chinese/) is loaded and reviewed.

CLASSIFICATION_PROMPT_V3_CHINESE = """You are a mental health text classifier analyzing social media posts for signs of depression.

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
- Using emotional words casually ("this is so depressing")
- Societal or political commentary not directed at the author's own mental state
- Third-person references to someone else's suffering

The post may be in Chinese script (Simplified or Traditional), Pinyin, or code-switched text.

Classify the following post. Respond with ONLY one label:
- Depressed
- Not Depressed

Post:
\"\"\"{post_text}\"\"\"

Classification:"""


# ── Prompt registry ─────────────────────────────────────────────────────────

PROMPTS = {
    "v1": CLASSIFICATION_PROMPT_V1,
    "v2": CLASSIFICATION_PROMPT_V2,
    "v3": CLASSIFICATION_PROMPT_V3,                       # Urdu-tuned few-shot
    "v3_arabic":      CLASSIFICATION_PROMPT_V3_ARABIC,
    "v3_chinese":     CLASSIFICATION_PROMPT_V3_CHINESE,
    "v3_exp2":        ATTRIBUTION_PROMPT_V3_EXP2,        # Urdu — attribution (explain existing label)
    "v3_arabic_exp2": ATTRIBUTION_PROMPT_V3_ARABIC_EXP2,  # Arabic — attribution
}

# ── Language → default prompt mapping ───────────────────────────────────────
# Used by the Phase 2 runner to auto-select the right prompt per language/experiment.

LANGUAGE_DEFAULT_PROMPTS = {
    "urdu":    "v3",
    "arabic":  "v3_arabic",
    "chinese": "v3_chinese",
}

LANGUAGE_DEFAULT_PROMPTS_EXP2 = {
    "urdu":    "v3_exp2",
    "arabic":  "v3_arabic_exp2",
    "chinese": "v3_chinese",   # placeholder — no exp2 version yet
}

# ── Active prompt ────────────────────────────────────────────────────────────
# Change this to switch which prompt the pipeline uses.

CLASSIFICATION_PROMPT = CLASSIFICATION_PROMPT_V2


def build_prompt(post_text: str, prompt_version: str = None) -> str:
    """Build the classification prompt for a given post."""
    template = PROMPTS.get(prompt_version, CLASSIFICATION_PROMPT) if prompt_version else CLASSIFICATION_PROMPT
    return template.format(post_text=post_text)
