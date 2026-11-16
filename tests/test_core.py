"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pathlib
import pytest

from app.core import (
    milestone_1, load_corpus, preprocess_corpus, CORPUS_PATH,
    milestone_2, NgramModel, build_ngram_model,
    milestone_3, suggest_next_words, generate_text, extract_context,
    milestone_4,
)


# ---------------------------------------------------------------------------
# Milestone 1: Curate an anonymized corpus of administrative medical text.
# ---------------------------------------------------------------------------
def test_milestone_1():
    """Milestone 1: corpus loads and preprocessing returns a usable token list."""

    # --- 1. The bundled corpus file must exist ---------------------------------
    assert CORPUS_PATH.exists(), (
        f"Corpus file missing at {CORPUS_PATH}. "
        "Create data/corpus.txt with anonymized medical notes."
    )

    # --- 2. load_corpus returns a non-empty string ----------------------------
    raw = load_corpus(CORPUS_PATH)
    assert isinstance(raw, str), "load_corpus() must return a str"
    assert len(raw) > 0, "Corpus must not be empty"

    # --- 3. preprocess_corpus produces a list of tokens -----------------------
    tokens = preprocess_corpus(raw)
    assert isinstance(tokens, list), "preprocess_corpus() must return a list"
    assert len(tokens) >= 100, (
        f"Expected at least 100 tokens, got {len(tokens)}. "
        "Corpus is too small for meaningful training."
    )

    # --- 4. All tokens are non-empty lowercase alphabetic strings -------------
    for tok in tokens:
        assert tok == tok.lower(), f"Token '{tok}' is not lowercase"
        assert tok.isalpha(), (
            f"Token '{tok}' contains non-alphabetic characters. "
            "Preprocessing should strip punctuation and digits."
        )
        assert len(tok) > 0, "Token list must not contain empty strings"

    # --- 5. No comment lines leaked into the token list -----------------------
    assert "#" not in tokens, "Comment marker '#' must be stripped during preprocessing"

    # --- 6. milestone_1() is the public entry point and returns same tokens ---
    result = milestone_1()
    assert isinstance(result, list)
    assert len(result) >= 100

    # --- 7. load_corpus raises FileNotFoundError for a missing path -----------
    with pytest.raises(FileNotFoundError):
        load_corpus(pathlib.Path("/nonexistent/path/corpus.txt"))

    # --- 8. Spot-check: common medical words appear in the corpus tokens ------
    medical_words = {"patient", "clinic", "medication", "assessment", "physician"}
    token_set = set(tokens)
    found = medical_words & token_set
    assert len(found) >= 3, (
        f"Expected common medical words in corpus but found only: {found}. "
        "Make sure the corpus contains administrative medical text."
    )


# ---------------------------------------------------------------------------
# Milestone 2: Implement the n-gram language model and apply smoothing.
# ---------------------------------------------------------------------------
def test_milestone_2():
    """Milestone 2: n-gram model trains correctly and smoothing works."""
    tokens = milestone_1()

    # --- 1. milestone_2() returns a trained NgramModel ---------------------
    model = milestone_2()
    assert isinstance(model, NgramModel), "milestone_2() must return an NgramModel"

    # --- 2. Vocabulary is populated ----------------------------------------
    assert len(model.vocab) > 0, "Model vocab must not be empty after training"
    assert "patient" in model.vocab, "'patient' should be in the medical corpus vocab"

    # --- 3. Count table is populated ----------------------------------------
    assert len(model.counts) > 0, "counts dict must be populated after training"

    # --- 4. Probabilities are valid floats in (0, 1] -----------------------
    # Known context from corpus
    context = ("patient",)
    prob = model.probability(context, "was")
    assert 0.0 < prob <= 1.0, f"Probability must be in (0, 1], got {prob}"

    # --- 5. Laplace smoothing: unseen word gets non-zero probability --------
    unseen_word = "xyzzy_unseen_word_zzzz"
    model.vocab.add(unseen_word)          # inject into vocab to test smoothing
    prob_unseen = model.probability(("patient",), unseen_word)
    assert prob_unseen > 0.0, (
        "Laplace smoothing must assign non-zero probability to unseen words"
    )

    # --- 6. Seen word more probable than unseen word -----------------------
    prob_seen = model.probability(("patient",), "was")
    assert prob_seen > prob_unseen, (
        "A word seen after 'patient' should be more probable than an unseen word"
    )

    # --- 7. All probabilities over vocab sum to approx 1.0 ----------------
    total = sum(model.probability(context, w) for w in model.vocab)
    assert abs(total - 1.0) < 1e-6, (
        f"Probabilities over full vocab must sum to 1.0, got {total:.6f}"
    )

    # --- 8. top_next_words returns sorted list of (word, prob) pairs -------
    top = model.top_next_words(("patient",), k=3)
    assert len(top) == 3, f"Expected 3 results, got {len(top)}"
    assert all(isinstance(w, str) and isinstance(p, float) for w, p in top)
    probs = [p for _, p in top]
    assert probs == sorted(probs, reverse=True), "Results must be sorted by probability descending"

    # --- 9. build_ngram_model works for n=3 (trigram) ----------------------
    trigram_model = build_ngram_model(tokens, n=3)
    assert trigram_model.n == 3
    assert len(trigram_model.vocab) > 0

    # --- 10. ValueError raised for n < 2 ------------------------------------
    with pytest.raises(ValueError):
        NgramModel(n=1)

    # --- 11. ValueError raised when corpus too short for given n -----------
    with pytest.raises(ValueError):
        tiny_model = NgramModel(n=5)
        tiny_model.train(["a", "b", "c"])


# ---------------------------------------------------------------------------
# Milestone 3: Develop a text generation feature that suggests words based on input.
# ---------------------------------------------------------------------------
def test_milestone_3():
    """Milestone 3: next-word suggestion and text generation work correctly."""
    model = milestone_2()

    # --- 1. milestone_3() returns a list of strings ------------------------
    suggestions = milestone_3("patient presented")
    assert isinstance(suggestions, list), "milestone_3() must return a list"
    assert len(suggestions) > 0, "must return at least one suggestion"
    assert all(isinstance(w, str) for w in suggestions), "all suggestions must be strings"

    # --- 2. suggest_next_words returns k results by default ----------------
    results = suggest_next_words(model, "patient presented to the clinic")
    assert isinstance(results, list)
    assert len(results) == 5, f"Default k=5 expected, got {len(results)}"

    # --- 3. All suggestions are known vocabulary words ---------------------
    for word in results:
        assert word in model.vocab, f"Suggested word '{word}' not in model vocab"

    # --- 4. Custom k is respected ------------------------------------------
    for k in (1, 3, 10):
        r = suggest_next_words(model, "patient", k=k)
        assert len(r) == k, f"Expected {k} results, got {len(r)}"

    # --- 5. Results are ordered most-probable first ------------------------
    # Re-score and confirm ordering matches model probabilities
    tokens_in = preprocess_corpus("patient presented to the clinic")
    context = extract_context(tokens_in, model.n)
    top_pairs = model.top_next_words(context, k=5)
    expected_order = [w for w, _ in top_pairs]
    assert results == expected_order, "suggest_next_words must return words in descending probability order"

    # --- 6. Works on mixed-case / punctuated raw input ---------------------
    raw_results = suggest_next_words(model, "Patient Presented To The Clinic!!!")
    assert isinstance(raw_results, list)
    assert len(raw_results) == 5

    # --- 7. Works when input is empty (graceful fallback) ------------------
    empty_results = suggest_next_words(model, "", k=3)
    assert isinstance(empty_results, list)
    assert len(empty_results) == 3

    # --- 8. extract_context trims to n-1 words correctly ------------------
    ctx = extract_context(["a", "b", "c", "d"], n=3)
    assert ctx == ("c", "d"), f"Expected ('c', 'd'), got {ctx}"

    # --- 9. extract_context handles tokens shorter than n-1 ---------------
    short_ctx = extract_context(["only"], n=3)
    assert short_ctx == ("only",), f"Expected ('only',), got {short_ctx}"

    # --- 10. generate_text produces a string with correct word count -------
    output = generate_text(model, "patient presented", num_words=5)
    assert isinstance(output, str), "generate_text() must return a str"
    output_tokens = output.split()
    # seed "patient presented" = 2 words + 5 generated = 7 total
    assert len(output_tokens) == 7, (
        f"Expected 7 tokens in generated text, got {len(output_tokens)}: '{output}'"
    )

    # --- 11. generate_text output starts with preprocessed seed words ------
    seed_tokens = preprocess_corpus("patient presented")
    assert output.startswith(" ".join(seed_tokens)), (
        "Generated text must begin with the seed words"
    )


# Milestone 4: Create a CLI for users to interact with the suggestion engine.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")
