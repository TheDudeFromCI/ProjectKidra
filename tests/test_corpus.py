"""JSON corpus grouping, tokenizer compatibility, and failure handling."""

import itertools
import json
from pathlib import Path

import pytest

from project_kidra.corpus.builder import CorpusBuilder, extract_samples
from project_kidra.corpus.file import CorpusFile
from project_kidra.corpus.tokenizer import BOS_TOKEN_STR, EOS_TOKEN_STR, FREE_TOKEN_STR, Tokenizer


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    """Use the bundled tokenizer without modifying IDs."""

    return Tokenizer()


def source(path: Path, value: object) -> Path:
    """Write a JSON fixture."""

    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_special_tokens(tokenizer: Tokenizer) -> None:
    """Bundled special token IDs and deterministic encoding remain available."""

    for text, token in (
        (BOS_TOKEN_STR, tokenizer.bos_token),
        (EOS_TOKEN_STR, tokenizer.eos_token),
        (FREE_TOKEN_STR, tokenizer.free_token),
    ):
        assert tokenizer.encode(text) == [token]
    assert len({tokenizer.bos_token, tokenizer.eos_token, tokenizer.free_token}) == 3
    assert Tokenizer().fingerprint == tokenizer.fingerprint
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_json_and_jsonl_preserve_text_and_groups(tmp_path: Path) -> None:
    """Unicode, whitespace, escaped newlines, and slot ordering are retained exactly."""

    samples = [{"windows": ["café\t\n中文", "other window"]}, {"windows": ["last"]}]
    json_path = source(tmp_path / "samples.json", samples)
    jsonl_path = tmp_path / "samples.jsonl"
    jsonl_path.write_text("\n" + "\n\n".join(json.dumps(s) for s in samples), encoding="utf-8")
    expected = [s["windows"] for s in samples]
    assert list(extract_samples(json_path)) == expected
    assert list(extract_samples(jsonl_path)) == expected
    assert list(extract_samples(source(tmp_path / "single.json", samples[0]))) == expected[:1]


@pytest.mark.parametrize(
    "sample",
    [
        None,
        "text",
        {},
        {"windows": []},
        {"windows": "text"},
        {"windows": [""]},
        {"windows": [1]},
        {"windows": ["x"], "typo": True},
    ],
)
def test_malformed_sample_rejected(tmp_path: Path, sample: object) -> None:
    """Fail explicitly instead of flattening or silently skipping malformed samples."""

    with pytest.raises(ValueError):
        list(extract_samples(source(tmp_path / "bad.json", sample)))


def test_find_sources(tmp_path: Path) -> None:
    """Discover both formats recursively, deduplicate, and ignore legacy text files."""

    nested = tmp_path / "nested"
    nested.mkdir()
    a = source(tmp_path / "a.json", {})
    b = source(nested / "b.jsonl", {})
    source(tmp_path / "old.txt", {})
    assert CorpusBuilder.find_sources([tmp_path, a], log=False) == sorted([a, b])
    assert CorpusBuilder.find_sources([tmp_path], "*.jsonl", log=False) == [b]
    with pytest.raises(FileNotFoundError):
        CorpusBuilder.find_sources([tmp_path / "missing"], log=False)


def test_roundtrip_groups_and_metadata(tmp_path: Path, tokenizer: Tokenizer) -> None:
    """Sampling returns complete multi-window groups; metadata counts all real targets."""

    texts = [["one", "two words"], ["three", "four", "five"]]
    path = source(tmp_path / "source.json", [{"windows": w} for w in texts])
    output = tmp_path / "corpus.kidra"
    metadata = CorpusBuilder(tokenizer).build([path], output)
    expected = [[tokenizer.encode_window(t) for t in w] for w in texts]
    assert metadata.samples == 2
    assert metadata.max_windows == 3
    assert metadata.num_tokens == sum(len(t) for w in expected for t in w)
    corpus = CorpusFile(output, preload=2)
    corpus.validate_tokenizer(tokenizer)
    assert corpus.metadata == metadata
    sampled = list(itertools.islice(corpus, 100))
    assert all(s in expected for s in sampled)
    assert all(s in sampled for s in expected)
    assert all(t[-1] == tokenizer.eos_token for w in sampled for t in w)
    tokenizer.fingerprint = "wrong"
    try:
        with pytest.raises(ValueError, match="tokenizer mismatch"):
            corpus.validate_tokenizer(tokenizer)
    finally:
        tokenizer.fingerprint = Tokenizer().fingerprint


def test_invalid_build_keeps_existing_output(tmp_path: Path, tokenizer: Tokenizer) -> None:
    """Malformed or empty input cannot overwrite a previously valid corpus."""

    output = tmp_path / "corpus.kidra"
    output.write_bytes(b"keep")
    invalid: list[object] = [[], {"windows": [5]}]
    for value in invalid:
        with pytest.raises(ValueError):
            CorpusBuilder(tokenizer).build([source(tmp_path / "bad.json", value)], output)
        assert output.read_bytes() == b"keep"
    with pytest.raises(ValueError):
        CorpusBuilder(tokenizer).build([], output)
    with pytest.raises(ValueError):
        CorpusBuilder(tokenizer).build([output], output)


def test_reject_legacy_corrupt_missing_and_invalid_preload(tmp_path: Path) -> None:
    """Fail immediately instead of blocking a background loader forever."""

    path = tmp_path / "old.npy"
    path.write_bytes(b"KIDRA" + bytes(30))
    with pytest.raises(ValueError, match="rebuild"):
        CorpusFile(path)
    with pytest.raises(ValueError, match="preload"):
        CorpusFile(path, preload=0)
    with pytest.raises(FileNotFoundError):
        CorpusFile(tmp_path / "missing")
