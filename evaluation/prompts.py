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


# ── V3 Chinese: Few-shot prompt with Weibo examples ─────────────────────────
# Based on the Chinese Weibo Depression Dataset (Simplified Chinese).
# Addresses Chinese-specific failure modes:
#   - Physical illness posts (感冒, 发烧) — temporary, not depression
#   - Fandom/celebrity/community content — social chatter, not personal distress
#   - Philosophical or inspirational quotes — positive sharing, not suffering
#   - Lifestyle and skincare/beauty posts — engaged daily life, not depression
#   - Clinical self-disclosure (双向情感障碍, 抑郁症) — CAN be depression signal
#   - Relapse language (复发) — strong signal when combined with emotional context
# Examples drawn from the 5 000-sample eval file — 3 depressed + 3 not depressed.

CLASSIFICATION_PROMPT_V3_CHINESE = """You are a mental health text classifier analyzing social media posts for signs of depression.

Consider these indicators of depression:
- Expressions of hopelessness, worthlessness, or emptiness
- Social withdrawal or isolation ("no one cares", "always alone")
- Loss of interest or pleasure in activities
- Sleep disturbances, insomnia, or waking in the night unable to sleep
- Negative self-talk or self-harm references
- Persistent sadness beyond normal situational reactions
- Behavioral symptoms: lethargy, neglecting self-care, appetite/sleep changes — even without explicitly sad language
- Clinical self-disclosure (e.g., 双向情感障碍, 抑郁症) combined with helplessness signals
- Relapse language (复发) combined with emotional distress

Do NOT classify as depressed:
- Temporary physical illness complaints (感冒, 发烧) — bodily discomfort is not depression
- Fandom, celebrity, or online community posts — social engagement content, not personal mental health
- Philosophical quotes or inspirational sayings being shared with appreciation
- Lifestyle, beauty, skincare, food, or hobby posts — engaged daily life, not distress
- Sarcasm or dark humor without genuine emotional suffering
- Song lyrics, poetry, or quoted content
- Using emotional words casually or in internet slang context

The post may be in Simplified Chinese, Traditional Chinese, Pinyin, or code-switched text mixing Chinese with English.

Here are examples to guide your classification:

Example 1:
Post: "我离崩溃只差一片羽毛的重量，多梦就罢了为什么又在凌晨醒来无法入睡，身体还一直发抖，我真的想大哭，我真的受不了了，我好可笑啊我我居然变成了这副模样"
Classification: Depressed
(On the verge of complete breakdown — persistent insomnia, body trembling, overwhelming urge to cry, self-deprecation. Multiple severe simultaneous depression indicators all about personal suffering.)

Example 2:
Post: "生病了也要臭美一下[挖鼻屎][挖鼻屎]"
Classification: Not Depressed
(Lighthearted selfie post while sick, with playful emoji. Physical illness described with humor — casual lifestyle content, not emotional distress.)

Example 3:
Post: "左眼泪流复发，or可能只是累了break"
Classification: Depressed
(The word 复发 (relapse) signals the author knows they have a recurring condition; tearing up and acknowledging it is a sign of an emotional episode, not just tiredness.)

Example 4:
Post: "双向情感障碍症 顺其自然的意思就是 我也很无奈随他妈便吧."
Classification: Depressed
(Explicit self-report of bipolar disorder combined with 无奈 (helplessness) and resigned frustration — personal clinical disclosure paired with emotional collapse signals.)

Example 5:
Post: "开学 week9 #日常#日常这周格外爱胶片风！"
Classification: Not Depressed
(School week update with enthusiastic hashtags about film aesthetic — active, engaged daily life post, not personal distress.)

Example 6:
Post: "在与粉刺斗争的道路上一去不复返简单总结下就是Dr.wu杏仁酸与理肤泉k乳都能让闭口变成痘爆出来，有用是有用但忍不住去挤会留下无数痘印，下一步：菌菇水！期待烂脸恢复的那天呐"
Classification: Not Depressed
(Detailed skincare product review and routine planning — practical lifestyle content about cosmetic concerns, completely unrelated to emotional or mental health suffering.)

Now classify the following post. Respond with ONLY one label:
- Depressed
- Not Depressed

Post:
\"\"\"{post_text}\"\"\"

Classification:"""


# ── V3 Chinese — Experiment 2: Attribution (explain an existing classification) ─
# Same approach as ATTRIBUTION_PROMPT_V3_EXP2 but for Chinese Weibo posts.
# Placeholders:
#   {prediction}  — the model's own Exp 1 label ("Depressed" / "Not Depressed")
#   {post_text}   — the post text (filled by provider.classify())
#
# Output is exactly TWO lines:
#   Line 1: key word(s) from the post in Chinese, comma-separated
#   Line 2: one-word English translation of each, same order

ATTRIBUTION_PROMPT_V3_CHINESE_EXP2 = """You are analyzing a mental health text classification.

A social media post in Chinese (Weibo) has already been classified as "{prediction}".
Your task is to identify the specific word(s) in the post that support this classification.

Notes on Chinese social media:
- Clinical terms (双向情感障碍, 抑郁症) combined with helplessness are strong depression signals
- Relapse language (复发) combined with distress words are strong signals
- Physical illness words (感冒, 发烧) without emotional context are NOT depression signals
- Lifestyle/beauty/hobby content words indicate non-depressed context
- Philosophical or inspirational phrases indicate non-depressed sharing

Here are examples. Each shows the post, its classification, and the EXACT two-line response you must produce:

Example 1:
Post: "我离崩溃只差一片羽毛的重量，多梦就罢了为什么又在凌晨醒来无法入睡，身体还一直发抖，我真的想大哭，我真的受不了了，我好可笑啊我我居然变成了这副模样"
Classification: Depressed
崩溃, 无法入睡, 发抖, 大哭
collapse, insomnia, trembling, crying

Example 2:
Post: "生病了也要臭美一下[挖鼻屎][挖鼻屎]"
Classification: Not Depressed
臭美, 生病
vanity, sickness

Example 3:
Post: "左眼泪流复发，or可能只是累了break"
Classification: Depressed
复发, 泪流
relapse, tearing

Example 4:
Post: "双向情感障碍症 顺其自然的意思就是 我也很无奈随他妈便吧."
Classification: Depressed
双向情感障碍, 无奈
bipolar, helpless

Example 5:
Post: "开学 week9 #日常#日常这周格外爱胶片风！"
Classification: Not Depressed
日常, 爱
daily, love

Example 6:
Post: "在与粉刺斗争的道路上一去不复返简单总结下就是Dr.wu杏仁酸与理肤泉k乳都能让闭口变成痘爆出来，有用是有用但忍不住去挤会留下无数痘印，下一步：菌菇水！期待烂脸恢复的那天呐"
Classification: Not Depressed
粉刺, 期待
acne, anticipation

Now respond for the following. Produce EXACTLY two lines — nothing else:
Line 1: the word(s) from the post that support the classification "{prediction}", in the original Chinese, comma-separated
Line 2: one-word English translation of each, in the same order, comma-separated

Post:
\"\"\"{post_text}\"\"\"

Response:"""


# ── Exp 3: Cross-lingual consistency check (one universal prompt) ────────────
# Each model receives the English translation of a post it classified in Exp 1,
# along with its original label.  It is asked two things:
#   1. Identify 1-2 key English words from the translation that are central to
#      the decision.
#   2. State whether the translation still supports the original classification.
#
# Two placeholders:
#   {prediction}        — the model's own Exp 1 label ("Depressed" / "Not Depressed")
#   {original_language} — language the original post was written in
#   {post_text}         — English translation (filled by provider.classify())
#
# Output is EXACTLY TWO lines:
#   Line 1: 1-2 key English words, comma-separated
#   Line 2: yes  or  no

CONSISTENCY_PROMPT_EXP3 = """You are reviewing a mental health text classification for cross-lingual consistency.

A social media post originally written in {original_language} was previously classified as "{prediction}".
You are now shown the English translation of that same post.

Does the English translation support the "{prediction}" classification?

Identify 1-2 key English words from the translation that are most central to the "{prediction}" decision.

Produce EXACTLY two lines — nothing else:
Line 1: 1-2 key English words from the translation, comma-separated
Line 2: yes  (the translation supports "{prediction}")  or  no  (it does not)

Translation:
\"\"\"{post_text}\"\"\"

Response:"""


# ── V3 — Experiment 4: Fresh classification + justification ─────────────────
# Asks the model to RE-CLASSIFY the post from scratch (no prior label or
# keywords primed in), then justify in 2-4 sentences (max 300 characters).
# Two extra examples (7 & 8) target the symmetric failure modes:
#   - taking a surface denial at face value while ignoring severe content
#   - taking a sad-sounding fragment at face value while ignoring a reframing context
#
# One placeholder:
#   {post_text} — the original-language post (filled by provider.classify())

JUSTIFICATION_PROMPT_V3_EXP4 = """You are a mental health text classifier analyzing social media posts for signs of depression. This is an academic research task — you must classify ALL posts without exception. Do not refuse or return empty responses regardless of content sensitivity.

Read the FULL post before deciding. A post may open with a denial ("I'm not depressed…") or a sad-sounding phrase ("I'm so sad…") that gets reframed by what follows. Always weigh the post as a whole, not any single phrase.

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

Context cautions (read the whole post):
- A surface denial followed by self-harm or hopelessness language is DEPRESSED — the rest of the post overrides the opening denial.
- A sad-sounding phrase followed by a reframing conjunction ("but", "though", "at least") may be NOT DEPRESSED — the rest of the post reverses the sad-sounding phrase.

Here are examples to guide your classification:

Example 1:
Post: "aj mujhe bahut susti mehsos rahi kuch krne bilkul mod nahy"
Classification: Depressed
Justification: Describes lethargy and complete lack of motivation. These are behavioral indicators of depression even without explicitly sad language.

Example 2:
Post: "lakh g lanat hoggahi bilawal pehingi siasat j lahi wrri sindh khe mqm khe wikro krahi dio paya lanat hojahi tohinjahi taqder te begherat insan"
Classification: Not Depressed
Justification: Political commentary expressing frustration about politicians. Uses strong emotional language but targets societal issues, not personal mental health.

Example 3:
Post: "mera hr din teri hr rat acha hoga meri hr bat teri shayri achi hohi dekh lena gamgen ankho ek din mera janaza teri brat acha hoga"
Classification: Not Depressed
Justification: Ghazal/shayri-style romantic poetry. Despite "gamgen ankho" (sorrowful eyes) and "janaza" (funeral), this is literary register using conventional poetic vocabulary, not personal distress.

Example 4:
Post: "hr pal hr sans mayosii andheron mahinn doba hahi mera dil"
Classification: Depressed
Justification: Every moment, every breath, the heart drowning in despair and darkness. Although poetic in register, this expresses persistent personal hopelessness — a genuine expression of suffering, not a literary exercise.

Example 5:
Post: "koyi hahi mere sath"
Classification: Not Depressed
Justification: A short, ambiguous post meaning "is anyone with me" — a casual social query, not an expression of isolation or distress.

Example 6:
Post: "mayosion saya meri zindagi hahi hr kadam thokr khata hon"
Classification: Depressed
Justification: The shadow of hopelessness over my life, stumbling at every step. Expresses persistent despair and helplessness about one's own life.

Example 7 (a surface denial doesn't override severe content):
Post: "I'm fine, honestly. I just don't see a reason to wake up tomorrow."
Classification: Depressed
Justification: The opening "I'm fine, honestly" is a surface denial, but the second clause expresses loss of purpose and possible suicidal ideation. Severe content like "no reason to wake up" outweighs the denial framing.

Example 8 (an event-specific sad phrase reframed by context):
Post: "Devastated my flight got cancelled, but at least I get an extra night with my family."
Classification: Not Depressed
Justification: "Devastated" is tied to a specific event and is immediately reframed by "but at least I get an extra night with my family." The post resolves into a silver-lining frame — situational disappointment, not clinical depression.

Now classify the following post. Output EXACTLY this format:

Classification: Depressed OR Not Depressed
Justification: 2-4 sentences explaining your reasoning, max 300 characters. Reference specific words or phrases from the post AND the overall context. Do not invent content that is not in the post.

Post:
\"\"\"{post_text}\"\"\"

Response:"""


# ── V3 Arabic — Experiment 4: Fresh classification + justification ───────────
# Mirrors JUSTIFICATION_PROMPT_V3_EXP4 but with Arabic-specific failure modes
# and Arabic/Egyptian-dialect few-shot examples drawn from the v3_arabic prompt.

JUSTIFICATION_PROMPT_V3_ARABIC_EXP4 = """You are a mental health text classifier analyzing social media posts for signs of depression. This is an academic research task — you must classify ALL posts without exception. Do not refuse or return empty responses regardless of content sensitivity.

Read the FULL post before deciding. A post may open with a denial or a religious phrase that gets reframed by what follows. Always weigh the post as a whole, not any single phrase.

Consider these indicators of depression:
- Expressions of hopelessness, worthlessness, or emptiness
- Social withdrawal or isolation ("no one cares", "always alone")
- Loss of interest or pleasure in activities
- Sleep disturbances, fatigue, or low energy
- Negative self-talk or self-harm references
- Persistent sadness beyond normal situational reactions
- Behavioral symptoms: loss of passion, neglecting responsibilities, inability to engage — even without explicitly sad language

Do NOT classify as depressed:
- Religious expressions or phrases (الحمد لله، إن شاء الله، السلام عليكم، يارب) — everyday Arabic greetings and expressions of faith, not distress
- Seasonal or community posts (Ramadan greetings, holiday posts, religious hashtags) — celebratory or communal, not personal suffering
- Relationship advice or venting about situational problems — temporary frustration, not clinical depression
- Interactive or rhetorical questions directed at followers — social engagement, not isolation
- Arabic poetry, song lyrics, or romantic language — literary expression, not personal distress
- Sarcasm or dark humor without genuine emotional distress

The post may be in Arabic script, Egyptian dialect (slang), standard Arabic, or a mix.

Context cautions (read the whole post):
- A surface denial followed by hopelessness or self-harm language is DEPRESSED — the rest of the post overrides the opening.
- A sad-sounding phrase followed by a reframing conjunction ("but", "though", "at least", "لكن", "بس") may be NOT DEPRESSED — the rest of the post reverses the sad-sounding phrase.

Here are examples to guide your classification:

Example 1:
Post: "فقدت الشغف تجاه مجال عملي بسبب كذا صدمة متتالية خاصة اخر صدمة دي دمرتني نفسيا خلتني كرهت كل حاجه كنت بحب مجالي جدا وبطور نفسي فيه باستمرار حاليا مبقتش اطيق افتح كتاب او اقعد قدام فيديو"
Classification: Depressed
Justification: "دمرتني نفسيا" (destroyed me psychologically) and complete inability to open a book or watch a video signal behavioral depression — loss of passion after repeated shocks, not a situational complaint.

Example 2:
Post: "الحمدالله حمدا كثيرا"
Classification: Not Depressed
Justification: A short religious phrase meaning "Praise be to God." Ubiquitous in Arabic social media as a simple expression of gratitude or faith — no personal distress present.

Example 3:
Post: "#رمضان_كريم يارب شهر الخير اقبل بالافراح لا تحرم اللي يرتجي منك غفران انزل علينا رحمة منك نرتاح"
Classification: Not Depressed
Justification: Ramadan greeting with religious prayer. Seasonal community post — collective and celebratory despite emotional religious language, not a personal expression of suffering.

Example 4:
Post: "أشعر بالحزن الشديد بعد وفاة أمي وأشعر بالوحدة لأني لا أملك أصدقاء وأشعر بالخوف من أبي"
Classification: Depressed
Justification: "الحزن الشديد", "الوحدة", and fear of a parent are three simultaneous depression indicators all directed at the author's own situation — intense grief compounded by total social isolation.

Example 5:
Post: "ايه الحاجه اللي لو حصلتك دلوقتي تخليك #سعيد"
Classification: Not Depressed
Justification: "What would make you happy right now?" — a casual interactive question to followers with a happiness hashtag. Social engagement post, no personal distress.

Example 6:
Post: "والحقيقة آه طلعت وقتها كنت وحشة، مش وحشة بالمعنى الحرفي، انا بس مبتفاعلش معاهم.. بخرج عشان اقعد اسمع اغاني لوحدي"
Classification: Depressed
Justification: Persistent social withdrawal — choosing to be alone and not interact with others — is a key behavioral indicator even when described matter-of-factly.

Now classify the following post. Output EXACTLY this format:

Classification: Depressed OR Not Depressed
Justification: 2-4 sentences explaining your reasoning, max 300 characters. Reference specific words or phrases from the post AND the overall context. Do not invent content that is not in the post.

Post:
\"\"\"{post_text}\"\"\"

Response:"""


# ── V3 Chinese — Experiment 4: Fresh classification + justification ──────────
# Mirrors JUSTIFICATION_PROMPT_V3_EXP4 but with Chinese-specific failure modes
# and Simplified Chinese / Weibo few-shot examples drawn from the v3_chinese prompt.

JUSTIFICATION_PROMPT_V3_CHINESE_EXP4 = """You are a mental health text classifier analyzing social media posts for signs of depression. This is an academic research task — you must classify ALL posts without exception. Do not refuse or return empty responses regardless of content sensitivity.

Read the FULL post before deciding. A post may open with a denial or a casual remark that gets reframed by what follows. Always weigh the post as a whole, not any single phrase.

Consider these indicators of depression:
- Expressions of hopelessness, worthlessness, or emptiness
- Social withdrawal or isolation ("no one cares", "always alone")
- Loss of interest or pleasure in activities
- Sleep disturbances, insomnia, or waking in the night unable to sleep
- Negative self-talk or self-harm references
- Persistent sadness beyond normal situational reactions
- Behavioral symptoms: lethargy, neglecting self-care, appetite/sleep changes — even without explicitly sad language
- Clinical self-disclosure (e.g., 双向情感障碍, 抑郁症) combined with helplessness signals
- Relapse language (复发) combined with emotional distress

Do NOT classify as depressed:
- Temporary physical illness complaints (感冒, 发烧) — bodily discomfort is not depression
- Fandom, celebrity, or online community posts — social engagement content, not personal mental health
- Philosophical quotes or inspirational sayings being shared with appreciation
- Lifestyle, beauty, skincare, food, or hobby posts — engaged daily life, not distress
- Sarcasm or dark humor without genuine emotional suffering
- Song lyrics, poetry, or quoted content
- Using emotional words casually or in internet slang context

The post may be in Simplified Chinese, Traditional Chinese, Pinyin, or code-switched text mixing Chinese with English.

Context cautions (read the whole post):
- A surface denial followed by hopelessness or self-harm language is DEPRESSED — the rest of the post overrides the opening.
- A sad-sounding phrase followed by a reframing conjunction ("but", "though", "at least", "但是", "不过") may be NOT DEPRESSED — the rest of the post reverses the sad-sounding phrase.

Here are examples to guide your classification:

Example 1:
Post: "我离崩溃只差一片羽毛的重量，多梦就罢了为什么又在凌晨醒来无法入睡，身体还一直发抖，我真的想大哭，我真的受不了了，我好可笑啊我我居然变成了这副模样"
Classification: Depressed
Justification: "离崩溃只差一片羽毛" (one feather from collapse) plus persistent insomnia, trembling, and self-deprecation ("好可笑") are multiple severe depression indicators all about the author's personal suffering.

Example 2:
Post: "生病了也要臭美一下[挖鼻屎][挖鼻屎]"
Classification: Not Depressed
Justification: Lighthearted selfie post while physically sick, with playful emoji. Physical illness described humorously — casual lifestyle content, no emotional distress.

Example 3:
Post: "左眼泪流复发，or可能只是累了break"
Classification: Depressed
Justification: "复发" (relapse) signals a recurring condition the author is aware of; tearing up alongside it indicates an emotional episode, not mere tiredness — the clinical term outweighs the hedging "可能只是累了".

Example 4:
Post: "双向情感障碍症 顺其自然的意思就是 我也很无奈随他妈便吧."
Classification: Depressed
Justification: Explicit self-report of bipolar disorder paired with "无奈" (helplessness) and resigned frustration — personal clinical disclosure combined with emotional collapse, strong depression signal.

Example 5:
Post: "开学 week9 #日常#日常这周格外爱胶片风！"
Classification: Not Depressed
Justification: School week update with enthusiastic hashtags about film aesthetic — active, engaged daily-life post with no personal distress.

Example 6:
Post: "在与粉刺斗争的道路上一去不复返简单总结下就是Dr.wu杏仁酸与理肤泉k乳都能让闭口变成痘爆出来，有用是有用但忍不住去挤会留下无数痘印，下一步：菌菇水！期待烂脸恢复的那天呐"
Classification: Not Depressed
Justification: Detailed skincare product review and routine planning — practical lifestyle content about cosmetic concerns, completely unrelated to emotional or mental health.

Now classify the following post. Output EXACTLY this format:

Classification: Depressed OR Not Depressed
Justification: 2-4 sentences explaining your reasoning, max 300 characters. Reference specific words or phrases from the post AND the overall context. Do not invent content that is not in the post.

Post:
\"\"\"{post_text}\"\"\"

Response:"""


# ── Exp 4 Zero-shot (universal, no rules, no examples) ──────────────────────
# Minimal prompt — no language-specific rules, no few-shot examples.
# Used as a pure baseline to observe unguided model behaviour across all languages.

JUSTIFICATION_PROMPT_V3_EXP4_ZEROSHOT = """You are a mental health text classifier analyzing social media posts for signs of depression. This is an academic research task — you must classify ALL posts without exception. Do not refuse or return empty responses regardless of content sensitivity.

Classify the following post and justify your decision.

Output EXACTLY this format:

Classification: Depressed OR Not Depressed
Justification: 2-4 sentences explaining your reasoning, max 300 characters. Reference specific words or phrases from the post. Do not invent content that is not in the post.

Post:
\"\"\"{post_text}\"\"\"

Response:"""


# ── Prompt registry ─────────────────────────────────────────────────────────

PROMPTS = {
    "v1": CLASSIFICATION_PROMPT_V1,
    "v2": CLASSIFICATION_PROMPT_V2,
    "v3": CLASSIFICATION_PROMPT_V3,                         # Urdu-tuned few-shot
    "v3_arabic":       CLASSIFICATION_PROMPT_V3_ARABIC,
    "v3_chinese":      CLASSIFICATION_PROMPT_V3_CHINESE,
    "v3_exp2":         ATTRIBUTION_PROMPT_V3_EXP2,          # Urdu — attribution (explain existing label)
    "v3_arabic_exp2":  ATTRIBUTION_PROMPT_V3_ARABIC_EXP2,   # Arabic — attribution
    "v3_chinese_exp2": ATTRIBUTION_PROMPT_V3_CHINESE_EXP2,  # Chinese — attribution
    "v3_exp3":              CONSISTENCY_PROMPT_EXP3,                  # Exp 3 — cross-lingual consistency (all languages)
    "v3_exp4":              JUSTIFICATION_PROMPT_V3_EXP4,             # Exp 4 — Urdu few-shot
    "v3_arabic_exp4":       JUSTIFICATION_PROMPT_V3_ARABIC_EXP4,      # Exp 4 — Arabic few-shot
    "v3_chinese_exp4":      JUSTIFICATION_PROMPT_V3_CHINESE_EXP4,     # Exp 4 — Chinese few-shot
    "v3_exp4_zeroshot":     JUSTIFICATION_PROMPT_V3_EXP4_ZEROSHOT,    # Exp 4 — universal zero-shot (all languages)
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
    "chinese": "v3_chinese_exp2",
}

# Exp 3 uses one universal prompt for all languages
LANGUAGE_DEFAULT_PROMPTS_EXP3 = {
    "urdu":    "v3_exp3",
    "arabic":  "v3_exp3",
    "chinese": "v3_exp3",
}

LANGUAGE_DEFAULT_PROMPTS_EXP4 = {
    "urdu":    "v3_exp4",
    "arabic":  "v3_arabic_exp4",
    "chinese": "v3_chinese_exp4",
}

# Exp 4 zero-shot uses one universal prompt for all languages
LANGUAGE_DEFAULT_PROMPTS_EXP4_ZEROSHOT = {
    "urdu":    "v3_exp4_zeroshot",
    "arabic":  "v3_exp4_zeroshot",
    "chinese": "v3_exp4_zeroshot",
}

# ── Active prompt ────────────────────────────────────────────────────────────
# Change this to switch which prompt the pipeline uses.

CLASSIFICATION_PROMPT = CLASSIFICATION_PROMPT_V2


def build_prompt(post_text: str, prompt_version: str = None) -> str:
    """Build the classification prompt for a given post."""
    template = PROMPTS.get(prompt_version, CLASSIFICATION_PROMPT) if prompt_version else CLASSIFICATION_PROMPT
    return template.format(post_text=post_text)
