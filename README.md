# Medical Notes Helper

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/2ybzv35kak/badge.svg)](https://ask.thili.ai/b/2ybzv35kak)

A CLI tool that helps clinic staff draft administrative sections of medical notes by generating **next-word suggestions** using an n-gram language model trained on anonymized medical text.

---

## What it does

- **Suggests** the most likely next words as you type a medical note phrase
- **Generates** full sentence continuations from a seed phrase
- Trained on a curated corpus of anonymized administrative medical text
- Uses **Laplace-smoothed bigram/trigram** language modelling — no API key needed, runs fully offline

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
patient was admitted for safety checklists were reviewed and documented per
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

## How well it works and where it falls short

### Model at a glance

| Metric | Value |
|---|---|
| Training tokens | 2,077 |
| Unique vocabulary | 687 words |
| Unique bigram contexts | 687 |
| Smoothing | Laplace (add-one) |
| Default n-gram order | 2 (bigram) |

### Where it works well ✅

- **Common administrative phrases** — phrases that appear frequently in the corpus get
  accurate, domain-appropriate suggestions:

  ```
  "discharge summary"       → indicates, included, and
  "vital signs were"        → reviewed, provided, discussed
  "patient presented"       → to, readings, electrocardiogram
  "assessment and plan"     → was, written, prescription
  ```

- **Short completions** — single next-word suggestions are reliable for phrases
  whose last word appears often in the training data.

- **Domain vocabulary** — the vocabulary is entirely medical/administrative, so
  suggestions never drift into unrelated topics.

- **Runs offline instantly** — no API call, no network, no latency. The model
  trains in under a second on the bundled corpus.

### Where it falls short ⚠️

- **Small corpus (2,077 tokens, 687 vocab words)** — the training data is too
  small to cover all real clinical phrases. When an input phrase is not seen
  during training, Laplace smoothing falls back to near-uniform probabilities,
  so suggestions become generic words like `"and"`, `"the"`, `"was"`:

  ```
  "xray showed pneumonia"  → readings, electrocardiogram, reporting  ← not relevant
  "blood sugar is"         → within, unremarkable, significant       ← generic
  ```

- **No long-range context** — a bigram model only remembers the **last 1 word**.
  It cannot connect a phrase like `"patient was admitted for observation"` back
  to `"patient"` three words earlier, so generated sentences can drift:

  ```
  "physical examination" → physical examination was completed and
                           documented per protocol was completed  ← loops
  ```

- **Repetition loops in generation** — greedy decoding picks the same most-likely
  word repeatedly once the model lands on a high-frequency context, producing
  repetitive output for longer generations.

- **No sentence boundaries** — the model treats the whole corpus as a flat token
  stream. It cannot tell where one sentence ends and another begins, so
  generated text may mix unrelated clause fragments.

- **Vocabulary limited to corpus words only** — any word not in `data/corpus.txt`
  gets zero raw count. Laplace smoothing prevents a zero probability but the
  score is still very low, so rare clinical terms will almost never be suggested.

### How to get better suggestions

| Goal | What to do |
|---|---|
| More context-aware results | Use `--ngram 3` (trigram model) |
| Broaden vocabulary | Add more sentences to `data/corpus.txt` and re-run |
| Reduce repetition | Lower `--words` count; use `--ngram 3` |
| Domain tuning | Add notes that match the specific clinic specialty |

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

- Display suggestions with their probability scores alongside each word
- Save and load the trained model to disk to avoid re-training on every run
- Add an interactive REPL mode that keeps context across multiple inputs
- Replace greedy decoding with top-k or temperature sampling to reduce repetition
- Extend the corpus with more specialty-specific administrative notes
