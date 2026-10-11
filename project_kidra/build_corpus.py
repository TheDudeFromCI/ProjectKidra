import argparse
from pathlib import Path

from project_kidra.corpus.builder import CorpusBuilder
from project_kidra.corpus.tokenizer import Tokenizer


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
    print(f"wrote {metadata.num_tokens} tokens from {len(sources)} files to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
