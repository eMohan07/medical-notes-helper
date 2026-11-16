"""Command-line entry point: `python -m app.cli --help`."""
import argparse
import sys

from app.core import (
    milestone_2,
    suggest_next_words,
    generate_text,
    milestone_4,
)


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def _cmd_hello(args: argparse.Namespace) -> None:  # noqa: ARG001
    print("Medical Notes Helper is ready. Open README.md for your milestones.")


def _cmd_suggest(args: argparse.Namespace) -> None:
    """Print next-word suggestions for the given input text."""
    model = milestone_2(n=args.ngram)
    suggestions = suggest_next_words(model, args.text, k=args.top)
    if not suggestions:
        print("No suggestions found.")
        return
    print(f"Suggestions after \"{args.text}\":")
    for i, word in enumerate(suggestions, 1):
        print(f"  {i}. {word}")


def _cmd_generate(args: argparse.Namespace) -> None:
    """Auto-complete a seed phrase by greedily generating more words."""
    model = milestone_2(n=args.ngram)
    result = generate_text(model, args.seed, num_words=args.words)
    print(result)


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medical-notes-helper",
        description=(
            "Clinic staff tool: draft administrative medical note sections "
            "with next-word suggestions powered by an n-gram language model."
        ),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # -- hello ---------------------------------------------------------------
    sub.add_parser("hello", help="Check that everything is wired up.")

    # -- suggest -------------------------------------------------------------
    p_suggest = sub.add_parser(
        "suggest",
        help="Show the top next-word suggestions for a given input phrase.",
    )
    p_suggest.add_argument(
        "text",
        help='Input text to continue, e.g. "patient presented to the"',
    )
    p_suggest.add_argument(
        "--top", "-k",
        type=int,
        default=5,
        metavar="K",
        help="Number of suggestions to show (default: 5).",
    )
    p_suggest.add_argument(
        "--ngram", "-n",
        type=int,
        default=2,
        metavar="N",
        help="N-gram order for the model (default: 2 = bigram).",
    )

    # -- generate ------------------------------------------------------------
    p_generate = sub.add_parser(
        "generate",
        help="Auto-complete a seed phrase by generating additional words.",
    )
    p_generate.add_argument(
        "seed",
        help='Starting phrase, e.g. "patient was admitted"',
    )
    p_generate.add_argument(
        "--words", "-w",
        type=int,
        default=10,
        metavar="W",
        help="Number of words to generate after the seed (default: 10).",
    )
    p_generate.add_argument(
        "--ngram", "-n",
        type=int,
        default=2,
        metavar="N",
        help="N-gram order for the model (default: 2 = bigram).",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "hello":
        _cmd_hello(args)
    elif args.command == "suggest":
        _cmd_suggest(args)
    elif args.command == "generate":
        _cmd_generate(args)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
