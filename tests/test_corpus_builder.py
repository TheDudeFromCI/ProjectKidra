from pathlib import Path

import pytest

from project_kidra.moe import TokenCorpus
from project_kidra.moe.corpus_builder import CorpusBuilder
from project_kidra.moe.corpus_file import CorpusFile
from project_kidra.tokenizer import Tokenizer


def _tokenizer(words: list[str] | None = None) -> Tokenizer:
    """Build a tiny tokenizer so tests don't depend on the full vocabulary."""

    tokenizer = Tokenizer()
    tokenizer.tokens = words or ["a", "b", "c", "d", "e"]
    return tokenizer


def test_find_sources_expands_directories_and_rejects_missing(tmp_path: Path) -> None:
    """Directories are searched recursively by pattern; missing inputs raise."""

    (tmp_path / "sub").mkdir()
    (tmp_path / "one.txt").write_text("a")
    (tmp_path / "sub" / "two.txt").write_text("b")
    (tmp_path / "skip.md").write_text("c")
    found = CorpusBuilder.find_sources([tmp_path])
    assert [p.name for p in found] == ["one.txt", "two.txt"]
    with pytest.raises(FileNotFoundError):
        CorpusBuilder.find_sources([tmp_path / "nope"])


def test_build_then_load_round_trips_and_counts_unknown(tmp_path: Path) -> None:
    """A built corpus loads back with identical ids and accurate metadata."""

    source = tmp_path / "in.txt"
    source.write_text("a b zzz c d e a b c")
    out = tmp_path / "corpus.npy"
    tokenizer = _tokenizer()
    metadata = CorpusBuilder(tokenizer).build([source], out)
    assert metadata.num_tokens == 9
    assert metadata.num_unknown == 1

    corpus = TokenCorpus.open(out, context_window=3, tokenizer=tokenizer)
    inputs, targets = corpus.next_batch()
    assert inputs.tolist() == [[0, 1, tokenizer.vocab_size() - 3]]
    assert targets.tolist() == [[1, tokenizer.vocab_size() - 3, 2]]


def test_load_rejects_changed_tokenizer_and_missing_metadata(tmp_path: Path) -> None:
    """Corpora built with a different vocabulary or lacking metadata are refused."""

    source = tmp_path / "in.txt"
    source.write_text("a b c d e")
    out = tmp_path / "corpus.npy"
    CorpusBuilder(_tokenizer()).build([source], out)
    with pytest.raises(ValueError, match="different tokenizer"):
        CorpusFile.load(out, _tokenizer(["x", "y"]))
    CorpusFile.metadata_path(out).unlink()
    with pytest.raises(ValueError, match="metadata"):
        CorpusFile.load(out, _tokenizer())


def test_build_requires_sources(tmp_path: Path) -> None:
    """Building from no files is an error."""

    with pytest.raises(ValueError):
        CorpusBuilder(_tokenizer()).build([], tmp_path / "c.npy")
