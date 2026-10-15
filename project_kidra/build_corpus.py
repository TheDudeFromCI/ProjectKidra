import argparse
from pathlib import Path

from project_kidra.corpus.builder import CorpusBuilder
from project_kidra.corpus.tokenizer import Tokenizer


def main() -> int:
    """Tokenize JSON/JSONL multi-window samples into an indexed corpus."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="JSON/JSONL files or directories")
    parser.add_argument("--output", type=Path, default=Path("corpus.kidra"))
    parser.add_argument("--pattern", default=None, help="glob used inside directories")
    parser.add_argument(
        "--pretrained", type=Path, help="local Qwen directory with tokenizer.json and config.json"
    )
    args = parser.parse_args()

    builder = CorpusBuilder(Tokenizer(args.pretrained))
    sources = builder.find_sources(args.inputs, args.pattern)
    metadata = builder.build(sources, args.output)
    print(f"wrote {metadata.num_tokens} tokens from {len(sources)} files to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
