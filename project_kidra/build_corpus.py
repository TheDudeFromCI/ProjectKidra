import argparse
from pathlib import Path

from project_kidra.moe.corpus_builder import CorpusBuilder
from project_kidra.tokenizer import Tokenizer


def main() -> int:
    """Tokenize text files or directories into a prebuilt corpus (.npy + .json)."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="text files or directories")
    parser.add_argument("--output", type=Path, default=Path("corpus.npy"))
    parser.add_argument("--pattern", default="*.txt", help="glob used inside directories")
    args = parser.parse_args()

    builder = CorpusBuilder(Tokenizer())
    sources = builder.find_sources(args.inputs, args.pattern)
    metadata = builder.build(sources, args.output)
    unknown = metadata.num_unknown / max(metadata.num_tokens, 1)
    print(
        f"wrote {metadata.num_tokens} tokens from {len(sources)} files to {args.output} "
        f"({unknown:.1%} unknown)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
