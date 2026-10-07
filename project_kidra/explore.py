import argparse
from pathlib import Path

from project_kidra.corpus import Tokenizer
from project_kidra.explorer import (
    Browser,
    ResultRenderer,
    SimilarityIndex,
    load_embeddings,
)


def main() -> int:
    """Browse learned word vectors of a checkpoint interactively."""

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, nargs="?", default=Path("kidra-continuous.pt")
    )
    parser.add_argument("-k", type=int, default=10, help="results to show")
    args = parser.parse_args()

    tokenizer = Tokenizer.default_vocab()
    embeddings = load_embeddings(args.checkpoint)
    if embeddings.shape[0] != tokenizer.vocab_size():
        raise SystemExit("checkpoint vocabulary size does not match the tokenizer")
    tokens = [tokenizer.get_token(i) for i in range(tokenizer.vocab_size())]
    Browser(ResultRenderer(SimilarityIndex(tokens, embeddings), args.k)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
