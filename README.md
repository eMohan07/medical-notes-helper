# Medical Notes Helper

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/2ybzv35kak/badge.svg)](https://ask.thili.ai/b/2ybzv35kak)

The Medical Notes Helper is a CLI tool designed for clinic staff to streamline the drafting of administrative medical notes. By leveraging an n-gram language model trained on anonymized medical text, it provides intelligent next-word suggestions, enhancing consistency and efficiency in the healthcare domain.

## What you are building

**Goal.** You will build a command-line interface (CLI) tool that helps clinic staff draft administrative sections of medical notes by generating next-word suggestions, ensuring consistency and reducing manual typing. This tool will demonstrate the ability to generate text from a language model, providing context-aware suggestions for medical note drafting.

**Done when.** You will have a functional CLI tool that takes user input and provides next-word suggestions based on a trained n-gram language model. The model will be built from a curated corpus of administrative medical text and incorporate smoothing techniques.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Curate an anonymized corpus of administrative medical text.**  
      Hint: Text preprocessing for corpus preparation
- [ ] **2. Implement the n-gram language model and apply smoothing.**  
      Hint: N-gram model implementation and smoothing techniques
- [ ] **3. Develop a text generation feature that suggests words based on input.**  
      Hint: Next-word prediction logic
- [ ] **4. Create a CLI for users to interact with the suggestion engine.**  
      Hint: Command-line argument parsing

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow the user to specify the 'N' value for the n-gram model at runtime.
- Implement a feature to display the top N suggestions, rather than just a single next word.
- Add functionality to save and load the trained language model to avoid re-training on every use.
