# Keyword evaluation buckets

Each `{model}_keyword_evaluation` cell uses one of the following categorical labels. Empty cells mean the model produced no keywords for that row.

| Label | Meaning |
|---|---|
| `LITERAL_READ` | Model took figurative language (sarcasm, metaphor, irony) at face value. |
| `SURFACE_TOPIC` | Picked topic nouns only; no affective/emotional content extracted. |
| `OPPOSITE_POLARITY` | Selected words that read as the opposite sentiment of the post. |
| `OFF_TOPIC` | Keywords don't relate to the post (junk / hallucinated). |
| `ACCURATE` | Keywords reasonably capture the post; the misclassification is elsewhere. |
| `NEGATION_MISSED` | Failed to incorporate an explicit negation or qualifier (e.g., "not by choice"). |
| `MALFORMED` | Output structure broken (paragraph instead of list, "Line 2:" prefix, etc.). |
