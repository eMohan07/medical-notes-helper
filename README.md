# Medical Notes Helper

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/2ybzv35kak/badge.svg)](https://ask.thili.ai/b/2ybzv35kak)

A CLI tool that helps clinic staff draft administrative sections of medical notes by generating **next-word suggestions** using an n-gram language model trained on anonymized medical text.

---

## What it does

- **Suggests** the most likely next words as you type a medical note phrase
- **Generates** full sentence continuations from a seed phrase
- Trained on a curated corpus of anonymized administrative medical text
- Uses **Laplace-smoothed bigram/trigram** language modelling — no API key needed

---

## Milestones

- [x] **1. Curate an anonymized corpus of administrative medical text.**
      Text preprocessing: lowercase, strip punctuation, tokenize
- [x] **2. Implement the n-gram language model and apply smoothing.**
      N-gram model with Laplace (add-one) smoothing
- [x] **3. Develop a text generation feature that suggests words based on input.**
      Next-word prediction and greedy auto-complete
- [x] **4. Create a CLI for users to interact with the suggestion engine.**
      `suggest` and `generate` sub-commands with argument parsing

---

## Setup

You need **Python 3.12** and **git**.

### 1 — Create a virtual environment and install dependencies

**macOS / Linux**
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Copy the settings file

```bash
cp .env.example .env        # macOS / Linux
```
```powershell
Copy-Item .env.example .env  # Windows PowerShell
```

Open `.env` and choose **one** model block (see comments inside).  
No model key is required to run the CLI — the n-gram model is built locally from the bundled corpus.

### 3 — Verify the install

```bash
python -m app.cli hello
```
Expected output:
```
Medical Notes Helper is ready. Open README.md for your milestones.
```

### 4 — Run the tests

```bash
pytest
```
Expected output:
```
7 passed in 0.3s
```

---

## Usage

### `suggest` — next-word suggestions

```bash
python -m app.cli suggest "patient presented to the"
```
```
Suggestions after "patient presented to the":
  1. patient
  2. procedure
  3. operative
  4. referral
  5. admission
```

**Options**

| Flag | Default | Description |
|---|---|---|
| `--top K` / `-k K` | `5` | Number of suggestions to show |
| `--ngram N` / `-n N` | `2` | N-gram order (2 = bigram, 3 = trigram) |

```bash
# Show top 3 suggestions
python -m app.cli suggest "assessment and plan" --top 3

# Use a trigram model for more context
python -m app.cli suggest "patient was admitted for" --ngram 3 --top 5
```

---

### `generate` — auto-complete a phrase

```bash
python -m app.cli generate "patient was admitted"
```
```
patient was admitted for safety was completed and documented per surgical
```

**Options**

| Flag | Default | Description |
|---|---|---|
| `--words W` / `-w W` | `10` | Number of words to generate |
| `--ngram N` / `-n N` | `2` | N-gram order |

```bash
# Generate 15 words
python -m app.cli generate "discharge summary indicates" --words 15

# Use trigram model
python -m app.cli generate "patient was admitted" --ngram 3 --words 10
```

---

### Help

```bash
python -m app.cli --help
python -m app.cli suggest --help
python -m app.cli generate --help
```

---

## Project structure

```
medical-notes-helper/
├── app/
│   ├── cli.py        # CLI entry point (suggest, generate commands)
│   ├── core.py       # Corpus loading, preprocessing, NgramModel, prediction
│   ├── llm.py        # LLM chat helper (future use)
│   └── storage.py    # Simple JSON key-value store
├── data/
│   └── corpus.txt    # Anonymized administrative medical text corpus
├── tests/
│   ├── test_core.py  # Milestone tests (all 4 milestones)
│   └── test_smoke.py # Plumbing / smoke tests
├── .env.example      # Template for model settings
├── requirements.txt
└── README.md
```

---

## Settings and secrets

Model settings live in `.env`, which is git-ignored. **Never commit `.env` or paste a key into code.**  
If a key ever reaches GitHub, treat it as leaked and rotate it immediately.

---

## Stretch ideas

- Allow the user to specify the `N` value for the n-gram model at runtime *(already supported via `--ngram`!)*
- Display suggestions with their probabilities alongside each word
- Save and load the trained model to disk to avoid re-training on every run
- Add an interactive REPL mode that keeps context across multiple inputs
