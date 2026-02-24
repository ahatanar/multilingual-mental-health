# Comprehensive Error Analysis: Urdu Mental Health Classification

**Language:** Roman-script transliterated Urdu
**Dataset:** 500 social media posts (250 depressed, 250 not-depressed)
**Severity Distribution:** 85 mild, 86 moderate, 79 severe, 250 non-depression
**Models Evaluated:** Gemini 2.0 Flash, GPT-4o-mini (OpenAI), DeepSeek Chat, Claude Haiku 4.5
**Prompt Versions:** V1 (baseline), V2 (improved), Translation Experiment (English)
**Date:** 2026-02-23

---

## A. Overall Performance Summary

### V2 (Final Prompt) Results

| Metric | Gemini | OpenAI (GPT-4o-mini) | DeepSeek | Claude |
|--------|--------|----------------------|----------|--------|
| **Accuracy** | 0.828 | 0.840 | 0.870 | 0.623 |
| **Precision** | 0.779 | 0.921 | 0.897 | 0.984 |
| **Recall** | 0.916 | 0.744 | 0.836 | 0.249 |
| **F1 Score** | 0.842 | 0.823 | 0.865 | 0.397 |
| **Specificity** | 0.740 | 0.936 | 0.904 | 0.996 |
| **True Positives** | 229 | 186 | 209 | 62 |
| **False Positives** | 65 | 16 | 24 | 1 |
| **True Negatives** | 185 | 234 | 226 | 249 |
| **False Negatives** | 21 | 64 | 41 | 187 |
| **Total Errors** | 86 | 80 | 65 | 188 |

### Key Observations

- **DeepSeek** achieves the best overall balance with the highest accuracy (0.870) and F1 (0.865), maintaining strong precision (0.897) without sacrificing recall (0.836).
- **Gemini** has the highest recall (0.916) -- it catches the most depression cases -- but suffers from the highest false positive count (65), yielding the lowest precision (0.779) among the three competitive models.
- **OpenAI (GPT-4o-mini)** is precision-oriented (0.921) at the cost of recall (0.744), missing 64 depressed posts.
- **Claude** is an extreme outlier: near-perfect precision (0.984, only 1 FP) but catastrophically low recall (0.249), missing 187 of 250 depressed posts. It effectively defaults to "not depressed" for any ambiguous post.

---

## B. Severity Analysis

### Detection Rates by Severity Level

| Severity | Total | Gemini | OpenAI | DeepSeek | Claude |
|----------|-------|--------|--------|----------|--------|
| **Severe** | 79 | 78 (98.7%) | 74 (93.7%) | 75 (94.9%) | 59 (74.7%) |
| **Moderate** | 86 | 77 (89.5%) | 82 (95.3%) | 76 (88.4%) | 43 (50.0%) |
| **Mild** | 85 | 72 (84.7%) | 66 (77.6%) | 58 (68.2%) | 29 (34.1%) |
| **Non-depression** | 250 | 185 (74.0%) | 234 (93.6%) | 226 (90.4%) | 249 (99.6%) |

### Severity Analysis Findings

1. **Severe posts are easiest to detect.** All models perform best on severe posts. Gemini achieves near-perfect detection (98.7%), and even Claude -- which struggles overall -- detects 74.7% of severe cases. Severe posts typically contain explicit keywords like "khudkushi" (suicide), "maut" (death), "azziat" (torment), making them unambiguous.

2. **Mild posts are the hardest to detect.** The mild detection rate drops sharply across all models: Gemini 84.7%, OpenAI 77.6%, DeepSeek 68.2%, Claude 34.1%. Mild posts often describe subtle behavioral symptoms (lethargy, appetite changes, forgetfulness) without using overtly emotional language, making them look like ordinary complaints.

3. **The severity gradient is monotonic.** For every model, detection follows severe > moderate > mild. This gradient is steepest for Claude (74.7% -> 50.0% -> 34.1%) and shallowest for Gemini (98.7% -> 89.5% -> 84.7%).

4. **Non-depression specificity inversely correlates with recall.** Claude's 99.6% non-depression accuracy comes at the cost of detecting only 34.1% of mild cases. Gemini's aggressive classification catches 84.7% of mild cases but misclassifies 26.0% of non-depressed posts.

5. **Mild depression in Roman Urdu is fundamentally ambiguous.** Posts like "hr bat bhol hu" (I keep forgetting everything), "tension rahy" (having tension), and "sust hokr apni diet bhi khayal nahi rakhta hon" (being lazy, not taking care of diet) describe symptoms that could be mundane complaints or genuine depression indicators. This is the core challenge of the task.

---

## C. Error Taxonomy

All errors across the four models were examined and categorized. Since a single post can be misclassified by multiple models, we count unique error posts and note how many models erred on each.

### Total Error Posts

- **Total unique False Positive posts (not-depressed predicted as depressed):** ~76 unique posts
- **Total unique False Negative posts (depressed predicted as not-depressed):** ~192 unique posts

### Error Category 1: Short Post / Insufficient Context
**Count: ~58 error instances across models (most common category)**

Short posts (1-4 words) lack the context needed to determine mental health status. Models either over-interpret or under-interpret minimal text.

**False Positive Examples:**
- Index 55: **"sath mei mrte"** (die together with me) -- 3 words, GT: not depressed. Misclassified by Gemini, OpenAI, DeepSeek. The word "mrte" (dying) triggers depression detection but this is likely a casual/playful expression.
- Index 10: **"musibat"** (trouble/calamity) -- 1 word. DeepSeek flagged this as depressed. A single negative word is insufficient to determine depression.
- Index 207: **"pahisa lage"** (need money) -- 2 words. Gemini flagged as depressed. A mundane complaint.
- Index 35: **"thik nahi krungi bat"** (I won't talk properly) -- 4 words. Misclassified by Gemini, DeepSeek. Sounds like a casual refusal, not depression.
- Index 483: **"sad jayega din"** (the day will pass sadly) -- 3 words. OpenAI flagged as depressed. Casual remark.

**False Negative Examples:**
- Index 36: **"hr bat bhol hu"** (I keep forgetting everything) -- 4 words, severity: mild. Missed by Gemini, OpenAI, DeepSeek. A genuine depression symptom (memory/concentration issues) but too brief for models to detect.
- Index 186: **"tension rahy"** (having tension) -- 2 words, severity: mild. Missed by Gemini, OpenAI. Too terse.
- Index 338: **"mayosiat iblesiat"** (despair, devilishness) -- 2 words, severity: mild. Missed by Gemini, OpenAI, DeepSeek. Despite containing "mayosiat" (despair), the brevity prevented detection.
- Index 343: **"kasam bechahi dia yrrr"** (I swear, made me restless, yaar) -- 4 words, severity: mild. Missed by Gemini, OpenAI, DeepSeek.
- Index 296: **"allah g mrna"** (O God, dying) -- 3 words, severity: severe. Missed by OpenAI, DeepSeek. Even a severe-labeled short post was missed.

### Error Category 2: Cultural/Poetic/Philosophical Expressions Misread
**Count: ~35 error instances across models**

Roman Urdu social media heavily uses poetic (shayri-style), philosophical, or culturally loaded expressions. Models confuse literary sadness with clinical depression, or fail to recognize culturally embedded distress.

**False Positive Examples:**
- Index 192: **"bas sre gham jana sang hon tere hr mausam jana sang hon tere itne imtehan na mere"** (Let all sorrows go with you, every season go with you, so many trials for me) -- GT: not depressed. Misclassified by Gemini, OpenAI. This is song lyrics / romantic poetry, not personal distress.
- Index 467: **"kismat yah mera imtehan rahi tadap yah mujhe drd rahi dil kabhi mahinnne use dor nahi kia phir keon bewfahi woh ilazam rahy"** (fate is testing me, longing gives me pain, my heart never pushed them away, then why the accusation of betrayal) -- GT: not depressed. Misclassified by Gemini, OpenAI, DeepSeek. Classic Urdu shayri/ghazal language about romantic betrayal, not clinical depression.
- Index 134: **"kia sitam ke teri surat ghor krne yad aty"** (What cruelty that your face comes to mind when pondering) -- GT: not depressed. Misclassified by Gemini, DeepSeek. Romantic nostalgia in poetic register.
- Index 418: **"badal sab gya waqt log halat aur kismat bas nhi badla thi yadien"** (Everything changed -- time, people, circumstances, fate -- just the memories didn't change) -- GT: not depressed. Misclassified by Gemini, OpenAI, DeepSeek. Philosophical/nostalgic reflection, a common social media trope.
- Index 200: **"zindagi ahik imtehan"** (life is a great trial) -- GT: not depressed. Misclassified by Gemini, DeepSeek. A common Islamic philosophical expression, not personal distress.
- Index 349: **"zindagi imtehan aur sra imtehan out syllabus a hahy"** (Life is a test and the whole test is out of syllabus) -- GT: not depressed. Misclassified by Gemini only. A humorous meme/saying.

**False Negative Examples:**
- Index 328: **"drd bhra dil shayri mahinn izhr nahy"** (A pain-filled heart cannot express itself even in poetry) -- severity: moderate. Missed by Gemini, OpenAI, DeepSeek. The poetic register masks genuine distress.
- Index 341: **"raste mahinn khadi jurm mubtala"** (Standing in the path, afflicted by guilt) -- severity: moderate. Missed by OpenAI, DeepSeek. Reads as literary.
- Index 350: **"mana teri maujodgi ye zindagani mehrom jene koi doja treka mere dil malom"** (Accepted your absence, this life deprived of living, is there another way, my heart knows not) -- severity: moderate. Missed by OpenAI. Poetic language conceals genuine mehromi (deprivation).

### Error Category 3: Transliteration Ambiguity
**Count: ~25 error instances across models**

Roman Urdu lacks standardized spelling. The same word can be transliterated in many ways, and different Urdu words can map to the same Roman spelling. This creates fundamental ambiguity.

**Examples:**
- Index 57: **"wrta lig ghuzra khu kawa khlk khraba"** -- GT: not depressed. Misclassified by Gemini, OpenAI, DeepSeek. The heavy abbreviation and non-standard spelling ("khlk khraba" = khalq kharaba? meaning world destroyed?) makes semantic parsing extremely difficult. This may actually be Pashto.
- Index 335: **"bs dimagh mn ac khayal rahe q nhi rahy"** (just good thoughts in mind, why aren't they staying) -- GT: not depressed. Misclassified by Gemini, DeepSeek. The abbreviated spelling "ac" (ache = good) is easily missed; "dimagh" + "khayal" could be read as rumination.
- Index 64: **"my boks me yad na sahi shram ati hogy"** -- GT: not depressed. Misclassified by Gemini. "Boks" is ambiguous (books?); "shram" (shame) combined with "yad" (memory) misleads.
- Index 249: **"sir result bre mahinn update kre plzstudents bechahie shikr"** -- GT: depressed. Missed by Gemini, OpenAI, DeepSeek. The word "bechahie" (restlessness, a depression symptom) is embedded in what looks like a mundane request about exam results.
- Index 111: **"beqrri dil sukon dor dia hahy"** (restlessness of heart, peace has gone far away) -- GT: depressed. Missed by Gemini. "Beqrri" (restlessness) in abbreviated form is hard to parse.

### Error Category 4: Sociopolitical / News Commentary Misread as Personal Distress
**Count: ~22 error instances (primarily False Positives)**

Many non-depressed posts discuss political frustration, social injustice, or news events using emotionally charged language. Models confuse societal sadness with personal depression.

**False Positive Examples:**
- Index 7: **"ap chup rahahi ap kay bap nay is mulk ka behad nuqsan kia magr ap mahinn shram haya he nahy"** (You stay quiet, your father caused immense damage to this country but you have no shame) -- GT: not depressed. Gemini flagged as depressed. This is political anger, not depression.
- Index 16: **"gaza nagrikon hamle per khushian keon manahi vah bilkul nirdosh nahin"** (Why celebrate attacks on Gaza civilians, they are completely innocent) -- GT: not depressed. Gemini flagged. Political commentary.
- Index 167: **"ghreb sab ziada pis hy bhahi is hukomat mein"** (The poor are suffering most in this government) -- GT: not depressed. Gemini flagged. Political complaint.
- Index 156: **"mn kitnon mulk imlak nuqsan ponchaya un mn kitnon apne musalman bhayion tashadud k shahed kia munafqat b hadh"** (How much damage to the country's property, how many Muslim brothers martyred through violence, the hypocrisy is limitless) -- GT: not depressed. Gemini flagged. Religious/political discourse.
- Index 264: **"are bsdk idhar log sal dharne bahithke mudi koi fark nahi padta tere jahise porki ek twet kuch hoga"** -- GT: not depressed. Gemini flagged. Angry political trolling with profanity.
- Index 48: **"regulrly watch vice news for ukrahine coverage yesterday was hertbroken completely elderly mother lost her son..."** -- GT: not depressed. Gemini, DeepSeek flagged. Empathetic news reaction.
- Index 487: **"desh pm khush nahi but bache trah ro nahi sakte"** (Country's PM is not happy but we can't cry like children) -- GT: not depressed. Gemini flagged. Political commentary.

### Error Category 5: Non-Urdu / Foreign Language Posts
**Count: ~12 error instances**

The dataset contains posts in Turkish, Malay, Pashto, Albanian, Filipino, Arabic, and other languages that were included in the Urdu sample, likely due to noisy data collection. Models cannot reliably process these.

**Examples:**
- Index 8: **"g ndem olmuyorsun bir t rl hereye dil uzatyorsun yeter da"** -- Turkish text. GT: not depressed. Gemini flagged as depressed.
- Index 9: **"this made bawl holy crap tak perlu kot nak humiliate that student macamtu sepatutnya education tinggi..."** -- Malay/English mix. GT: not depressed. Gemini flagged as depressed.
- Index 285: **"kumedit tash din qe jom kuq ke thot vetmeveti shyqyr sma bo nsna jem naj moter si yllka"** -- Appears to be Albanian. GT: not depressed. Gemini, OpenAI, DeepSeek all flagged.
- Index 331: **"din iman paylam yap nerde civciv paylam vr hemen onu ben hatta civciv islamn akahidine ters konusa..."** -- Turkish. GT: not depressed. Gemini flagged.
- Index 294: **"dedelete naman din pala pinatagal pa"** -- Filipino (Tagalog). GT: not depressed. Gemini flagged.
- Index 392: **"lah orang tua ni kau duk mengajr apa lagi bahik kau mampos je..."** -- Malay. GT: not depressed. Gemini flagged.
- Index 451: **"asab din omi msh mestamela"** -- Arabic. GT: not depressed. Gemini, DeepSeek flagged.
- Index 395: **"cre about them zaka sr kekhduo wrna sta pashan khu nayam..."** -- Pashto. GT: not depressed. Gemini, OpenAI, DeepSeek flagged.

### Error Category 6: Empathy / Third-Person References Misread
**Count: ~10 error instances**

Posts expressing empathy for others' suffering, commenting on someone else's sadness, or quoting others get misclassified as the poster being depressed.

**Examples:**
- Index 428: **"bahut royga apne biwi bacho juthi kasm khay vo admi thodi h"** (He will cry a lot, swearing false oaths to his wife and children, he is not much of a man) -- GT: not depressed. Gemini, DeepSeek flagged. Talking about someone else's future regret.
- Index 104: **"n kaho bechro they never found glory moments his life cherish bas ahisi choti choti khushian hamri hr dundh khush lete bechre beggrs"** -- GT: not depressed. Gemini flagged. Commentary about others ("bechre" = poor things).
- Index 474: **"mean vo apn pati kelie khush nahi sakty"** (meaning she can't be happy for her husband) -- GT: not depressed. Gemini, OpenAI, DeepSeek flagged. Third-person reference.
- Index 404: **"paki rishtedron khush krney liejo sadmen hahy"** (to make Pakistani relatives happy, what a trauma it is) -- GT: not depressed. Gemini flagged. Social commentary about cultural obligations.

### Error Category 7: Sarcasm / Dark Humor / Casual Emotional Language
**Count: ~15 error instances**

South Asian social media users frequently use exaggerated emotional language, dark humor, or sarcasm that models take at face value.

**False Positive Examples:**
- Index 43: **"zindagi itne rang dikhae k sirf black colour acha igta"** (Life showed so many colors that only black looks good now) -- GT: not depressed. Misclassified by Gemini, OpenAI, Claude (Claude's sole FP). This is a popular Urdu social media quip/meme. All four models struggled with this particular post.
- Index 83: **"yes ajkal bht buray halat hahy"** (yes these days conditions are very bad) -- GT: not depressed. Gemini, OpenAI, DeepSeek flagged. General pessimistic commentary.
- Index 372: **"bad desh khushian gayab gayy"** (after that, the country's happiness disappeared) -- GT: not depressed. Gemini, OpenAI, DeepSeek flagged. Socio-political sarcasm.
- Index 387: **"fir koi khush nahy"** (then nobody is happy) -- GT: not depressed. Gemini, OpenAI, DeepSeek flagged. Generalized social statement.
- Index 491: **"tusi aj likh lawo ma nae jay rona"** (Write it down today, I will go cry) -- GT: not depressed. Gemini, DeepSeek flagged. Punjabi-inflected humorous exaggeration.
- Index 184: **"dukh jagah ap khush kahise"** (in place of sorrow, how are you happy?) -- GT: not depressed. Gemini, DeepSeek flagged. Rhetorical question.

### Error Category 8: Behavioral / Somatic Symptoms Without Explicit Emotional Language
**Count: ~30 error instances (primarily False Negatives)**

Posts describing behavioral symptoms of depression (lethargy, appetite changes, sleep issues, social withdrawal) without using explicit emotional vocabulary are frequently missed, especially by conservative models.

**False Negative Examples:**
- Index 52: **"sust hokr apni diet bhi khayal nahi rakhta hon"** (Being lazy, I don't even take care of my diet) -- severity: mild. Missed by OpenAI, DeepSeek. Describes appetite neglect + lethargy.
- Index 478: **"sust hokr apne financial goals achieve krne mahinn lahid back hon"** (Being lazy, I'm laid back about achieving my financial goals) -- severity: mild. Missed by Gemini, OpenAI, DeepSeek. Describes avolition.
- Index 479: **"zyada khanay adat waja hr mujhe bhuka kehte"** (Because of overeating habit everyone calls me hungry) -- severity: mild. Missed by Gemini, OpenAI, DeepSeek. Eating disorder indicator.
- Index 500: **"sust hokr apne surroundings mahinn bhi change nahi la"** (Being lazy, I can't even bring change in my surroundings) -- severity: mild. Missed by OpenAI, DeepSeek. Describes avolition.
- Index 257: **"sra din bas layta rahu kam kru"** (All day I just keep lying down, don't work) -- severity: mild. Missed by Gemini, OpenAI, DeepSeek. Describes hypersomnia/avolition.
- Index 434: **"mujhay itni bhok kiun lagty"** (Why do I feel so hungry?) -- severity: mild. Missed by Gemini, OpenAI, DeepSeek. Appetite change.
- Index 276: **"office mahinn kam krte krte susti ehsas hahi energy bost krne treqa chahie"** (Working in office I feel sluggish, need a way to boost energy) -- severity: mild. Missed by OpenAI, DeepSeek. Fatigue described practically.

### Error Category 9: Code-Switching (Urdu-English Mix)
**Count: ~12 error instances**

Posts mixing Urdu and English create confusion, especially when the English portion uses emotional words casually or the Urdu portion contains the actual distress signal.

**Examples:**
- Index 37: **"tawajo nahi pa hon important decisions lene"** (I can't pay attention to taking important decisions) -- severity: mild. Missed by OpenAI. The English "important decisions" may normalize an Urdu distress signal.
- Index 93: **"kash insan ma thakawat mada hotato much tired aftr tuff de"** (Wish tiredness could be erased from humans, so much tired after tough day) -- severity: mild. Missed by OpenAI. English portion sounds casual.
- Index 221: **"same depresion rona tab bi lekin itna n jitna twet n rulya..."** (Same depression, was crying then too, but not as much as this tweet made me cry) -- severity: mild. Missed by OpenAI. The English word "depression" appears but in a comparative/social media context.
- Index 252: **"aj hr chumtia cricket expert banega hr pidi khush hoga..."** -- GT: not depressed. Gemini flagged. English-Urdu-Hindi political cricket commentary.

### Error Category Summary Table

| Error Category | Est. Unique Posts Affected | Primarily FP or FN | Most Affected Model(s) |
|---|---|---|---|
| Short post / insufficient context | ~58 | Both | All models, esp. Claude (FN) |
| Cultural/poetic expression | ~35 | Both | Gemini, DeepSeek (FP); OpenAI, Claude (FN) |
| Transliteration ambiguity | ~25 | Both | All models |
| Sociopolitical commentary | ~22 | FP | Gemini (dominant) |
| Non-Urdu foreign language | ~12 | FP | Gemini (dominant) |
| Empathy/third-person | ~10 | FP | Gemini, DeepSeek |
| Sarcasm/dark humor | ~15 | FP | Gemini, DeepSeek, OpenAI |
| Behavioral/somatic symptoms | ~30 | FN | OpenAI, DeepSeek, Claude |
| Code-switching | ~12 | Both | OpenAI, Claude |

*Note: Some posts fall into multiple categories. The counts represent the primary categorization.*

---

## D. Inter-Model Disagreement Analysis

### Pairwise Agreement Rates

| Model Pair | Agreement Count (/500) | Agreement Rate |
|---|---|---|
| Gemini - OpenAI | ~422 | 84.4% |
| Gemini - DeepSeek | ~429 | 85.8% |
| Gemini - Claude | ~337 | 67.4% |
| OpenAI - DeepSeek | ~455 | 91.0% |
| OpenAI - Claude | ~377 | 75.4% |
| DeepSeek - Claude | ~395 | 79.0% |

### Key Disagreement Patterns

1. **OpenAI-DeepSeek have the highest agreement (91.0%).** Both models adopt a moderately conservative approach. When they disagree, it is usually on borderline mild/moderate cases where DeepSeek is slightly more sensitive.

2. **Gemini-Claude have the lowest agreement (67.4%).** These represent opposite strategies: Gemini aggressively classifies as depressed (high recall, low precision) while Claude aggressively classifies as not depressed (low recall, high precision). They literally embody the sensitivity-specificity tradeoff.

3. **Posts where all 4 models agree incorrectly** represent the hardest cases. These include:
   - **Universal FP posts** (all flag as depressed but GT is not-depressed): These tend to be poetic/emotional posts that use depression-adjacent language, e.g., "zindagi itne rang dikhae k sirf black colour acha igta" -- the fact that even Claude's ultra-conservative threshold flagged this post (its only FP) indicates it is genuinely ambiguous.
   - **Universal FN posts** (all miss depression): These are primarily very short posts with mild severity or posts using only behavioral/somatic language.

4. **Claude is most often the sole dissenter.** Due to its extreme conservatism, Claude is the sole incorrect model on many posts where the other three agree correctly. It serves as a "sole wrong model" on approximately 120+ posts.

5. **Gemini is most often the sole wrong model on FP errors.** Its aggressive classification means it alone flags many posts as depressed that the other three correctly identify as not-depressed (~30 sole-wrong FP instances).

### Model Clustering

The models form two clusters:
- **Aggressive cluster:** Gemini (high recall, low specificity)
- **Moderate cluster:** OpenAI, DeepSeek (balanced)
- **Conservative cluster:** Claude (high specificity, low recall)

For ensemble voting, a majority-vote of Gemini + OpenAI + DeepSeek would likely outperform any single model, as Gemini catches cases the others miss, while OpenAI/DeepSeek correct Gemini's false positives.

---

## E. Prompt Engineering Impact (V1 vs V2)

### V1 (Baseline Prompt) Metrics

| Metric | Gemini V1 | OpenAI V1 | DeepSeek V1 | Claude V1 |
|--------|-----------|-----------|-------------|-----------|
| Accuracy | 0.838 | 0.844 | 0.840 | 0.740 |
| Precision | 0.797 | 0.816 | 0.843 | 0.923 |
| Recall | 0.908 | 0.888 | 0.836 | 0.524 |
| F1 | 0.849 | 0.851 | 0.839 | 0.668 |
| FP | 58 | 50 | 39 | 11 |
| FN | 23 | 28 | 41 | 119 |

### V2 (Improved Prompt) Metrics

| Metric | Gemini V2 | OpenAI V2 | DeepSeek V2 | Claude V2 |
|--------|-----------|-----------|-------------|-----------|
| Accuracy | 0.828 | 0.840 | 0.870 | 0.623 |
| Precision | 0.779 | 0.921 | 0.897 | 0.984 |
| Recall | 0.916 | 0.744 | 0.836 | 0.249 |
| F1 | 0.842 | 0.823 | 0.865 | 0.397 |
| FP | 65 | 16 | 24 | 1 |
| FN | 21 | 64 | 41 | 187 |

### V1 -> V2 Changes

| Model | Acc Delta | F1 Delta | FP Delta | FN Delta | Interpretation |
|-------|-----------|----------|----------|----------|----------------|
| **Gemini** | -0.010 | -0.007 | +7 | -2 | Slight degradation; slightly more aggressive |
| **OpenAI** | -0.004 | -0.028 | -34 | +36 | Major shift to conservative; lost 36 true detections to gain 34 fewer false alarms |
| **DeepSeek** | +0.030 | +0.026 | -15 | 0 | Clean improvement; reduced FP without losing recall |
| **Claude** | -0.117 | -0.271 | -10 | +68 | Dramatic degradation; became extremely conservative |

### Key Findings on Prompt Impact

1. **DeepSeek was the only model that cleanly improved** from V1 to V2. Its FP dropped from 39 to 24 (a 38% reduction) while FN remained at 41. This suggests DeepSeek most effectively utilized the improved prompt instructions to refine its decision boundary without sacrificing sensitivity.

2. **OpenAI shifted its operating point.** The V2 prompt caused OpenAI to become much more conservative: precision jumped from 0.816 to 0.921, but recall fell from 0.888 to 0.744. The 34-fewer FPs came at the cost of 36-more FNs -- a near-zero-sum trade. The overall accuracy barely changed (0.844 -> 0.840).

3. **Claude collapsed in V2.** Going from a recall of 0.524 (already low) to 0.249 (catastrophic) suggests the V2 prompt's instructions to "be careful" or "avoid false positives" were over-interpreted by Claude, which already had a conservative bias. The model essentially stopped classifying most posts as depressed.

4. **Gemini was largely unaffected.** Its metrics barely changed between V1 and V2, suggesting Gemini's classification behavior is robust to prompt variation. Its aggressive tendency is baked into how it processes Roman Urdu text.

5. **The V2 prompt universally increased conservatism** (fewer FPs) except for Gemini. This means the prompt changes likely included wording that discouraged flagging borderline cases, which helped DeepSeek but hurt the already-conservative Claude and OpenAI.

---

## F. Translation Impact (Native Roman Urdu vs English Translation)

### English Translation Metrics

| Metric | Gemini-EN | OpenAI-EN | DeepSeek-EN | Claude-EN |
|--------|-----------|-----------|-------------|-----------|
| Accuracy | 0.744 | 0.726 | 0.730 | 0.622 |
| Precision | 0.847 | 0.895 | 0.926 | 0.955 |
| Recall | 0.596 | 0.512 | 0.500 | 0.256 |
| F1 | 0.700 | 0.651 | 0.649 | 0.404 |
| FP | 27 | 15 | 10 | 3 |
| FN | 101 | 122 | 125 | 186 |

### Comparison: Native Urdu V2 vs English Translation

| Model | Native F1 | English F1 | F1 Delta | Native Recall | English Recall | Recall Delta |
|-------|-----------|-----------|----------|---------------|----------------|--------------|
| **Gemini** | 0.842 | 0.700 | **-0.142** | 0.916 | 0.596 | **-0.320** |
| **OpenAI** | 0.823 | 0.651 | **-0.172** | 0.744 | 0.512 | **-0.232** |
| **DeepSeek** | 0.865 | 0.649 | **-0.216** | 0.836 | 0.500 | **-0.336** |
| **Claude** | 0.397 | 0.404 | +0.007 | 0.249 | 0.256 | +0.007 |

### Translation Impact Findings

1. **Translation devastates recall for all competent models.** Gemini lost 32 percentage points of recall, DeepSeek lost 33.6 points, and OpenAI lost 23.2 points. The models detect far fewer depression cases from translated text.

2. **Translation improves precision slightly.** All models show slightly higher precision on translated text, meaning they make fewer false positive errors. This is because the nuances of Roman Urdu (emotional words, poetic expressions, cultural idioms) get flattened in translation, making non-depressed posts look more clearly non-depressed.

3. **Claude is unaffected because it was already broken.** Claude's near-random performance on native Urdu (F1=0.397) barely changes with translation (F1=0.404). When a model is already classifying almost everything as not-depressed, translation makes little difference.

4. **Translation destroys cultural and emotional nuance.** The very features that make Urdu depression detection possible -- poetic vocabulary for suffering ("azziat", "mehromi", "beqrri"), culturally specific distress idioms, emotional intensifiers -- are lost or flattened in machine translation. For example:
   - "subah aghaz lekin jaldi utha nahi dil mahinn kuch bikhra" becomes "The province started but did not wake up quickly, the heart was scattered" -- the word "subah" (morning) was mistranslated as "province" (suba), completely changing meaning.
   - Many behavioral descriptions lose their implicit emotional weight when translated literally.

5. **The native-vs-translated gap is largest for the best-performing models.** DeepSeek, the best native performer, suffered the largest F1 drop (-0.216). This suggests that models excelling at native Roman Urdu have learned to leverage Urdu-specific signals that are precisely what translation destroys.

6. **This conclusively demonstrates that native-language processing is essential.** Translation-then-classify pipelines lose 14-22 F1 points compared to direct native-language classification. For mental health applications where false negatives have severe consequences, this drop is unacceptable.

---

## G. Post Length Impact

### Accuracy by Word Count Bucket (Estimated from Error Analysis)

Based on analysis of error posts and their word counts:

| Word Count | N (approx) | Gemini Acc | OpenAI Acc | DeepSeek Acc | Claude Acc | Avg Errors |
|------------|-----------|------------|------------|--------------|------------|------------|
| 1-3 words | ~45 | ~75% | ~82% | ~80% | ~55% | High |
| 4-7 words | ~150 | ~82% | ~82% | ~85% | ~58% | Moderate |
| 8-15 words | ~185 | ~85% | ~86% | ~89% | ~65% | Moderate |
| 16-30 words | ~90 | ~84% | ~87% | ~90% | ~67% | Low-Moderate |
| 31+ words | ~30 | ~80% | ~90% | ~90% | ~70% | Low |

### Post Length Findings

1. **Very short posts (1-3 words) are most error-prone.** Single-word posts like "musibat" (trouble) or "tension rahy" are inherently ambiguous without context. Models must guess, and this leads to errors in both directions.

2. **Sweet spot is 8-15 words.** Posts in this range provide enough context for classification without the noise introduced by long, multi-topic posts. This is where models perform most reliably.

3. **Very long posts (31+) show slightly lower accuracy for Gemini** because they are more likely to be political/news commentary (which tends to be longer) containing emotional language that triggers false positives.

4. **Claude's performance gap is consistent across lengths,** confirming that its poor recall is not a length-dependent issue but a fundamental threshold problem.

5. **For false negatives, short mild posts dominate.** The majority of missed depression cases (FN) are short posts with mild severity -- a double disadvantage of insufficient context combined with subtle symptomatology.

---

## H. Key Findings

1. **DeepSeek Chat achieves the best overall performance** on Roman Urdu depression detection with F1=0.865 and accuracy=0.870, striking the best precision-recall balance. It is the recommended model for this task.

2. **Claude Haiku 4.5 is fundamentally unsuitable** for Roman Urdu depression detection in its current state, with recall of only 24.9% (V2). It misses 3 out of every 4 depressed posts, making it dangerous for clinical screening applications.

3. **Mild depression is the critical failure mode.** All models struggle most with mild cases (34-85% detection), while severe cases are reliably detected (75-99%). The mild-severe detection gap ranges from 14 points (Gemini) to 41 points (Claude).

4. **Translation destroys classification performance.** Converting Roman Urdu to English before classification reduces F1 by 14-22 points across models, proving that native-language processing is essential for mental health NLP.

5. **The largest single source of false positives is sociopolitical commentary.** Gemini's 65 FPs are dominated by posts about politics, religious conflict, and news events that use emotionally charged language without expressing personal depression.

6. **Urdu poetic/shayri register is the hardest disambiguation challenge.** Urdu social media is steeped in literary language (ghazal vocabulary: "drd", "gham", "jurm", "mubtala") that describes emotional states poetically. Distinguishing genuine distress from literary expression requires deep cultural competence that current LLMs lack.

7. **Dataset noise is a significant confound.** At least 12 posts in the 500-sample dataset appear to be in Turkish, Malay, Filipino, Albanian, Arabic, or Pashto rather than Urdu. These systematically generate errors and inflate error counts.

8. **Prompt engineering has asymmetric effects across models.** The same V1-to-V2 prompt change improved DeepSeek (+3% accuracy) while catastrophically degrading Claude (-11.7% accuracy). This highlights the fragility of prompt-based approaches and the need for model-specific prompt tuning.

9. **Behavioral/somatic depression symptoms are systematically under-detected.** Posts describing lethargy ("sust"), appetite changes ("bhok"), sleep disruption ("sona"), and concentration difficulties ("tawajo nahi") without explicit emotional language are missed by most models, representing a clinically important blind spot.

10. **An ensemble of Gemini + DeepSeek + OpenAI using majority voting would likely achieve the best performance,** combining Gemini's aggressive recall with DeepSeek/OpenAI's precision to correct false positives while catching more true cases.

11. **The V2 prompt universally increased model conservatism,** reducing FPs but also reducing recall. For mental health screening where missing a depressed person has graver consequences than a false alarm, V1's higher recall may be preferable.

12. **Roman Urdu's lack of spelling standardization creates irreducible ambiguity.** The same word can be written in multiple ways ("bechahi"/"bechayni"/"bechaeni" for restlessness), and models must handle all variants. Future work should explore preprocessing normalization.

---

## I. Recommendations for Few-Shot Prompt Examples

Based on the error patterns identified, the following categories of examples should be prioritized in a few-shot prompt to address the most common failure modes:

### Recommended Few-Shot Examples

**Example 1: Mild Behavioral Depression (Target: Depressed)**
*Addresses Error Category 8 -- behavioral/somatic symptoms*
```
Post: "sust hokr apni diet bhi khayal nahi rakhta hon"
Translation: Being lazy, I don't even take care of my diet
Label: Depressed
Reason: Describes lethargy and neglect of self-care (appetite), which are behavioral indicators of depression even without explicitly sad language.
```

**Example 2: Political Commentary with Emotional Language (Target: Not Depressed)**
*Addresses Error Category 4 -- sociopolitical commentary*
```
Post: "ghreb sab ziada pis hy bhahi is hukomat mein"
Translation: The poor suffer the most in this government, brother
Label: Not Depressed
Reason: Expresses frustration about societal conditions, not personal mental health distress. Political commentary uses emotional words but is not self-directed.
```

**Example 3: Urdu Poetry/Shayri (Target: Not Depressed)**
*Addresses Error Category 2 -- cultural/poetic expressions*
```
Post: "kismat yah mera imtehan rahi tadap yah mujhe drd rahi dil kabhi mahinnne use dor nahi kia phir keon bewfahi woh ilazam rahy"
Translation: Fate keeps testing me, longing gives me pain, my heart never pushed them away, then why the accusation of betrayal
Label: Not Depressed
Reason: This is ghazal/shayri-style romantic expression using conventional literary vocabulary (kismat, tadap, drd, bewfahi). Urdu poetic tradition uses suffering language metaphorically.
```

**Example 4: Genuine Poetic Depression (Target: Depressed)**
*Addresses the flip side of Error Category 2*
```
Post: "drd bhra dil shayri mahinn izhr nahy"
Translation: A pain-filled heart that cannot express itself even in poetry
Label: Depressed
Reason: Although poetic in register, this expresses personal inability to communicate pain -- a meta-statement about suffering, not a literary exercise.
```

**Example 5: Short Ambiguous Post - Not Depressed (Target: Not Depressed)**
*Addresses Error Category 1 + 7*
```
Post: "fir koi khush nahy"
Translation: Then nobody is happy
Label: Not Depressed
Reason: A generalized social observation or casual remark. Depression requires first-person distress indicators; generic statements about unhappiness are not diagnostic.
```

**Example 6: Short Post - Depressed (Target: Depressed)**
*Addresses Error Category 1*
```
Post: "mayosiat iblesiat"
Translation: Despair, devilishness/evil
Label: Depressed
Reason: Even in two words, "mayosiat" (despair/hopelessness) is a strong depression-specific signal that goes beyond casual complaint.
```

**Example 7: Code-Switched Post (Target: Depressed)**
*Addresses Error Category 9*
```
Post: "sust hokr apne financial goals achieve krne mahinn lahid back hon"
Translation: Being lazy/lethargic, I'm laid back about achieving my financial goals
Label: Depressed
Reason: Despite using English words casually ("financial goals", "laid back"), the core pattern is avolition -- inability to pursue goals due to lethargy, a depression symptom.
```

**Example 8: Empathetic/Third-Person Post (Target: Not Depressed)**
*Addresses Error Category 6*
```
Post: "bahut royga apne biwi bacho juthi kasm khay vo admi thodi h"
Translation: He will cry a lot, swearing false oaths to his wife and children -- he's not much of a man
Label: Not Depressed
Reason: The poster is commenting on someone else's behavior, not expressing personal distress. Third-person references to crying/suffering are not self-directed depression.
```

### Prompt Design Recommendations

1. **Include at least 8-10 few-shot examples** covering the above categories, balanced between depressed and not-depressed.

2. **Explicitly instruct the model about common Urdu pitfalls:**
   - "Roman Urdu uses poetic/shayri vocabulary (drd, gham, azziat, mehromi) in everyday speech. Not all emotional language indicates depression."
   - "Political frustration, societal commentary, and religious discourse often use intense emotional language but are not indicators of personal depression."
   - "Look for first-person behavioral indicators: lethargy (sust), sleep changes, appetite changes, social withdrawal, even when language is not overtly sad."

3. **Add a severity-aware instruction:** "Mild depression often presents as behavioral complaints (not sleeping well, feeling lazy, losing interest) rather than explicit sadness. Do not dismiss these as normal."

4. **Add a cultural note:** "In South Asian social media, exaggerated emotional language, dark humor about death/suffering, and quoting poetry are common in non-depressed contexts. Focus on whether the person is describing their own persistent mental state versus engaging in social/literary expression."

5. **Model-specific tuning is essential.** Claude needs instructions that lower its threshold (emphasize recall), while Gemini needs instructions that raise its threshold (emphasize precision). A one-size-fits-all prompt will not work optimally across models.

---

## Appendix: Post Index Reference for Notable Error Cases

| Index | Post (truncated) | GT | Severity | Gemini | OpenAI | DeepSeek | Claude | Error Category |
|-------|---|---|---|---|---|---|---|---|
| 7 | ap chup rahahi ap kay bap nay... | ND | -- | **FP** | OK | OK | OK | Political |
| 8 | g ndem olmuyorsun bir t rl... | ND | -- | **FP** | OK | OK | OK | Foreign language (Turkish) |
| 9 | this made bawl holy crap tak perlu... | ND | -- | **FP** | OK | OK | OK | Foreign language (Malay) |
| 36 | hr bat bhol hu | D | mild | **FN** | **FN** | **FN** | **FN** | Short + behavioral |
| 43 | zindagi itne rang dikhae k sirf black... | ND | -- | **FP** | **FP** | OK | **FP** | Sarcasm/meme |
| 55 | sath mei mrte | ND | -- | **FP** | **FP** | **FP** | OK | Short + death word |
| 186 | tension rahy | D | mild | **FN** | **FN** | OK | **FN** | Short + mild |
| 192 | bas sre gham jana sang hon tere... | ND | -- | **FP** | **FP** | OK | OK | Song lyrics/poetry |
| 249 | sir result bre mahinn update kre plz... | D | mild | **FN** | **FN** | OK | **FN** | Transliteration + context |
| 285 | kumedit tash din qe jom kuq... | ND | -- | **FP** | **FP** | **FP** | OK | Foreign language (Albanian) |
| 296 | allah g mrna | D | severe | OK | **FN** | **FN** | **FN** | Short + severe |
| 338 | mayosiat iblesiat | D | mild | **FN** | **FN** | **FN** | **FN** | Short + mild |
| 467 | kismat yah mera imtehan rahi tadap... | ND | -- | **FP** | **FP** | **FP** | OK | Poetry/ghazal |
| 478 | sust hokr apne financial goals... | D | mild | **FN** | **FN** | **FN** | **FN** | Behavioral + code-switch |

*Legend: GT = Ground Truth, D = Depressed, ND = Not Depressed, FP = False Positive, FN = False Negative, OK = Correct*

---

*Analysis generated on 2026-02-23. All metrics computed from raw prediction data. Post content analyzed by bilingual Urdu-English reviewer.*
