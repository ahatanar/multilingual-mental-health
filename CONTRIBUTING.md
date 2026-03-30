# Adding Models — Quick Guide

This project currently targets **3 online models** (Gemini, GPT-4o-mini, DeepSeek) and **3 local models** (Llama, Qwen, Mistral via LM Studio). This guide explains exactly what to change to run any of them or swap in something new.

---

## Running an existing model

### Online models (Gemini / GPT-4o-mini / DeepSeek / Claude)

1. **Get your API key** from the provider's developer portal.
2. **Add it to `.env`** (copy `.env.example` to `.env` if you haven't):
   ```
   GEMINI_API_KEY=your-key-here
   OPENAI_API_KEY=your-key-here
   DEEPSEEK_API_KEY=your-key-here
   CLAUDE_API_KEY=your-key-here
   ```
3. **Run the runner** — the model will appear in the interactive menu automatically:
   ```bash
   python scripts/phase2/runner.py
   ```

### Local models (Llama / Qwen / Mistral via LM Studio)

1. **Install LM Studio** — download from [lmstudio.ai](https://lmstudio.ai).
2. **Load the model** you want to run (Llama 3.2, Qwen 2.5, or Mistral 7B).
3. **Start the local server** in LM Studio (default port: `1234`).
4. **Copy the model identifier** shown in LM Studio's "Local Server" tab
   (e.g. `llama-3.2-3b-instruct`, `qwen2.5-7b-instruct`).
5. **Check the `default_model` value** in `scripts/phase2/runner.py` for your model key (`llama`, `qwen`, or `mistral`) and update it if the identifier doesn't match:
   ```python
   "llama": {"class": LMStudioProvider, ..., "default_model": "llama-3.2-3b-instruct"},
   ```
6. **Run the runner** — local models appear in the menu alongside online ones.
   No API key or `.env` change needed.

---

## Adding a brand new online model

### Step 1 — Create a provider class

Copy the closest existing provider in `models/` as a starting point:

```bash
cp models/openai_provider.py models/myprovider_provider.py
```

Edit `models/myprovider_provider.py`:
- Change the class name to `MyProviderProvider`.
- Update `__init__` to use the provider's SDK / base URL.
- Implement `_call_api(self, prompt: str) -> str` — send the prompt, return the raw text.

### Step 2 — Export the class

Add it to `models/__init__.py`:

```python
from .myprovider_provider import MyProviderProvider
```

### Step 3 — Add the API key mapping

In `config.py`, add an entry to `ENV_MAP`:

```python
ENV_MAP = {
    ...
    "myprovider": "MYPROVIDER_API_KEY",   # <-- add this
}
```

Then add the key to your `.env`:

```
MYPROVIDER_API_KEY=your-key-here
```

### Step 4 — Register the model in the runner

In `scripts/phase2/runner.py`, add an entry to the `MODELS` dict:

```python
MODELS = {
    ...
    "myprovider": {"class": MyProviderProvider, "name": "My Model Name",
                   "default_model": "model-id-string"},
}
```

That's it — the model will appear in the interactive menu on the next run.

---

## Adding a new local model

### Step 1 — Load the model in LM Studio and note its identifier

Start the local server in LM Studio and copy the exact model identifier string.

### Step 2 — Add the model to the runner

```python
MODELS = {
    ...
    "mylocal": {"class": LMStudioProvider, "name": "My Model (Local)",
                "default_model": "exact-model-id-from-lmstudio",
                "max_workers": 1, "delay": 0},
}
```

### Step 3 — Register it as a local provider (no API key needed)

In `config.py`, add the key to `LOCAL_PROVIDERS`:

```python
LOCAL_PROVIDERS = {"lmstudio", "llama", "qwen", "mistral", "mylocal"}
```

---

## File reference

| File | What to change |
|------|----------------|
| `.env` | Add API keys for new online models |
| `config.py` | `ENV_MAP` (online) or `LOCAL_PROVIDERS` (local) |
| `models/<name>_provider.py` | New provider class (online only) |
| `models/__init__.py` | Export the new class |
| `scripts/phase2/runner.py` | `MODELS` dict — one entry per model |