"""Tests for the corpus module: tokenizer, builder, and corpus file loading."""

import itertools
from pathlib import Path

import pytest

from project_kidra.corpus import builder as builder_module
from project_kidra.corpus.builder import (
    MAGIC_NUMBER,
    CorpusBuilder,
    extract_samples,
)
from project_kidra.corpus.file import CorpusFile
from project_kidra.corpus.tokenizer import (
    BOS_TOKEN_STR,
    EOS_TOKEN_STR,
    FREE_TOKEN_STR,
    Tokenizer,
)


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    """A real tokenizer loaded from the bundled assets."""

    return Tokenizer()


@pytest.fixture(autouse=True)
def small_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the corpus header so tests don't write ~120 MB files."""

    monkeypatch.setattr(builder_module, "HEADER_SIZE", (8 + 4) * 1024)


def wrap(text: str) -> str:
    """Wrap text in BOS/EOS markers to form one sample."""

    return f"{BOS_TOKEN_STR}{text}{EOS_TOKEN_STR}"


def write_source(path: Path, content: str) -> Path:
    """Write a UTF-8 source file and return its path."""

    path.write_text(content, encoding="utf-8")
    return path


# --- Tokenizer ---


class TestTokenizer:
    """Tests for the word-level tokenizer."""

    def test_special_tokens_are_distinct_and_in_vocab(self, tokenizer: Tokenizer) -> None:
        """BOS, EOS and free tokens must be unique IDs within the vocabulary."""

        ids = {tokenizer.bos_token, tokenizer.eos_token, tokenizer.free_token}
        assert len(ids) == 3
        assert all(0 <= i < tokenizer.vocab_size for i in ids)

    def test_special_token_strings_encode_to_single_ids(self, tokenizer: Tokenizer) -> None:
        """Special token strings should map to their dedicated IDs."""

        assert tokenizer.encode(BOS_TOKEN_STR) == [tokenizer.bos_token]
        assert tokenizer.encode(EOS_TOKEN_STR) == [tokenizer.eos_token]
        assert tokenizer.encode(FREE_TOKEN_STR) == [tokenizer.free_token]

    def test_fingerprint_is_stable_sha256(self, tokenizer: Tokenizer) -> None:
        """The fingerprint is a hex SHA-256 digest and is identical across instances."""

        assert len(tokenizer.fingerprint) == 64
        assert Tokenizer().fingerprint == tokenizer.fingerprint

    def test_encode_returns_valid_ids(self, tokenizer: Tokenizer) -> None:
        """Every encoded ID must lie within the vocabulary and fit in 3 bytes."""

        ids = tokenizer.encode(wrap("The quick brown fox jumps over the lazy dog."))
        assert ids[0] == tokenizer.bos_token
        assert ids[-1] == tokenizer.eos_token
        assert all(0 <= i < min(tokenizer.vocab_size, 2**24) for i in ids)

    def test_encode_is_deterministic(self, tokenizer: Tokenizer) -> None:
        """Encoding the same text twice yields the same IDs."""

        text = "Hello, world! 123"
        assert tokenizer.encode(text) == tokenizer.encode(text)

    def test_empty_string(self, tokenizer: Tokenizer) -> None:
        """Empty text encodes to nothing and decodes to an empty string."""

        assert tokenizer.encode("") == []
        assert tokenizer.decode([]) == ""

    def test_decode_contains_original_words(self, tokenizer: Tokenizer) -> None:
        """Decoding an encoded sentence should preserve its words."""

        decoded = tokenizer.decode(tokenizer.encode("the cat sat"))
        assert "cat" in decoded


# --- Non-ASCII filtering ---


class TestAsciiFiltering:
    """The builder strips everything outside printable ASCII (plus tab/newline)."""

    @staticmethod
    def extract(tmp_path: Path, content: str) -> list[str]:
        """Run sample extraction over the given content."""

        return list(extract_samples(write_source(tmp_path / "src.txt", content)))

    def test_non_ascii_characters_removed(self, tmp_path: Path) -> None:
        """Accented letters, CJK, and emoji are dropped from samples."""

        samples = self.extract(tmp_path, wrap("caf\u00e9 \u4e2d\u6587 \U0001f600 ok"))
        assert samples == [wrap("caf   ok")]

    def test_all_output_is_ascii(self, tmp_path: Path) -> None:
        """No character above 126 survives, whatever the input."""

        junk = "".join(chr(i) for i in range(128, 3000))
        (sample,) = self.extract(tmp_path, wrap("a" + junk + "b"))
        assert sample == wrap("ab")
        assert sample.isascii()

    def test_control_characters_removed_but_newline_kept(self, tmp_path: Path) -> None:
        """Control chars (e.g. NUL, ESC) are stripped; newline is preserved."""

        (sample,) = self.extract(tmp_path, wrap("a\x00b\x1bc\nd"))
        assert sample == wrap("abc\nd")

    def test_tab_becomes_two_spaces(self, tmp_path: Path) -> None:
        """Tabs are normalised to two spaces."""

        (sample,) = self.extract(tmp_path, wrap("a\tb"))
        assert sample == wrap("a  b")

    def test_del_character_removed(self, tmp_path: Path) -> None:
        """DEL (127) is not printable and must be removed."""

        (sample,) = self.extract(tmp_path, wrap("a\x7fb"))
        assert sample == wrap("ab")

    def test_invalid_utf8_bytes_do_not_crash(self, tmp_path: Path) -> None:
        """Undecodable bytes are replaced and then filtered out."""

        path = tmp_path / "bad.txt"
        path.write_bytes(BOS_TOKEN_STR.encode() + b"a\xff\xfeb" + EOS_TOKEN_STR.encode())
        samples = list(extract_samples(path))
        assert len(samples) == 1
        assert samples[0].isascii()
        assert samples[0] == wrap("ab")

    def test_encoded_filtered_text_never_contains_unknown_chars(
        self, tokenizer: Tokenizer, tmp_path: Path
    ) -> None:
        """Filtered text round-trips through the tokenizer without non-ASCII output."""

        (sample,) = self.extract(tmp_path, wrap("na\u00efve r\u00e9sum\u00e9"))
        assert tokenizer.decode(tokenizer.encode(sample)).isascii()


# --- Sample extraction ---


class TestExtractSamples:
    """Tests for splitting source files into BOS/EOS delimited samples."""

    def test_multiple_samples(self, tmp_path: Path) -> None:
        """Each BOS..EOS span is yielded separately."""

        content = wrap("one") + wrap("two") + wrap("three")
        samples = list(extract_samples(write_source(tmp_path / "a.txt", content)))
        assert samples == [wrap("one"), wrap("two"), wrap("three")]

    def test_text_outside_markers_ignored(self, tmp_path: Path) -> None:
        """Text between samples is discarded."""

        content = "junk" + wrap("keep") + "more junk" + wrap("also")
        samples = list(extract_samples(write_source(tmp_path / "a.txt", content)))
        assert samples == [wrap("keep"), wrap("also")]

    def test_unterminated_sample_dropped(self, tmp_path: Path) -> None:
        """A sample missing its EOS marker is not yielded."""

        content = wrap("done") + BOS_TOKEN_STR + "never ends"
        samples = list(extract_samples(write_source(tmp_path / "a.txt", content)))
        assert samples == [wrap("done")]

    def test_no_markers_yields_nothing(self, tmp_path: Path) -> None:
        """A file without markers produces no samples."""

        assert list(extract_samples(write_source(tmp_path / "a.txt", "plain"))) == []

    def test_sample_spanning_chunks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Samples and markers split across read chunks are still found."""

        monkeypatch.setattr(builder_module, "CHUNK_SIZE", 7)
        content = wrap("a fairly long sample") + wrap("second")
        samples = list(extract_samples(write_source(tmp_path / "a.txt", content)))
        assert samples == [wrap("a fairly long sample"), wrap("second")]


# --- Builder ---


class TestFindSources:
    """Tests for CorpusBuilder.find_sources."""

    def test_directory_is_searched_recursively(self, tmp_path: Path) -> None:
        """Matching files in nested directories are found; others ignored."""

        (tmp_path / "sub").mkdir()
        a = write_source(tmp_path / "a.txt", "x")
        b = write_source(tmp_path / "sub" / "b.txt", "x")
        write_source(tmp_path / "c.md", "x")

        assert CorpusBuilder.find_sources([tmp_path], log=False) == sorted([a, b])

    def test_explicit_file_and_dedup(self, tmp_path: Path) -> None:
        """Files passed directly are included once even if also in a directory."""

        a = write_source(tmp_path / "a.txt", "x")
        assert CorpusBuilder.find_sources([a, tmp_path], log=False) == [a]

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        """Nonexistent inputs raise FileNotFoundError."""

        with pytest.raises(FileNotFoundError):
            CorpusBuilder.find_sources([tmp_path / "nope.txt"], log=False)

    def test_custom_pattern(self, tmp_path: Path) -> None:
        """The glob pattern is honoured."""

        write_source(tmp_path / "a.txt", "x")
        md = write_source(tmp_path / "b.md", "x")
        assert CorpusBuilder.find_sources([tmp_path], pattern="*.md", log=False) == [md]


class TestBuild:
    """Tests for CorpusBuilder.build."""

    def test_build_without_sources_raises(self, tokenizer: Tokenizer, tmp_path: Path) -> None:
        """An empty source list is rejected."""

        with pytest.raises(ValueError):
            CorpusBuilder(tokenizer).build([], tmp_path / "out.bin")

    def test_metadata_and_header(self, tokenizer: Tokenizer, tmp_path: Path) -> None:
        """Metadata reflects the tokenizer and counts; the file starts with the magic number."""

        src = write_source(tmp_path / "a.txt", wrap("hello world") + wrap("goodbye"))
        out = tmp_path / "out.bin"

        meta = CorpusBuilder(tokenizer).build([src], out)

        expected = sum(len(tokenizer.encode(s)) for s in (wrap("hello world"), wrap("goodbye")))
        assert meta.samples == 2
        assert meta.num_tokens == expected
        assert meta.vocab_size == tokenizer.vocab_size
        assert meta.tokenizer_fingerprint == tokenizer.fingerprint
        assert meta.sources == [str(src)]
        assert out.read_bytes()[: len(MAGIC_NUMBER)] == MAGIC_NUMBER

    def test_non_ascii_filtered_in_built_corpus(self, tokenizer: Tokenizer, tmp_path: Path) -> None:
        """Tokens in the built corpus equal those of the ASCII-only text."""

        src = write_source(tmp_path / "a.txt", wrap("caf\u00e9 \U0001f600 time"))
        out = tmp_path / "out.bin"
        CorpusBuilder(tokenizer).build([src], out)

        sample = next(iter(CorpusFile(out, preload=1)))
        assert sample == tokenizer.encode(wrap("caf  time"))


# --- CorpusFile ---


class TestCorpusFile:
    """Tests for loading and sampling from a built corpus."""

    @staticmethod
    def build(tokenizer: Tokenizer, tmp_path: Path, texts: list[str]) -> Path:
        """Build a corpus containing one sample per text."""

        src = write_source(tmp_path / "src.txt", "".join(wrap(t) for t in texts))
        out = tmp_path / "out.bin"
        CorpusBuilder(tokenizer).build([src], out)
        return out

    def test_invalid_magic_number_raises(self, tmp_path: Path) -> None:
        """Files without the magic header are rejected."""

        bad = tmp_path / "bad.bin"
        bad.write_bytes(b"NOPE!" + bytes(64))
        with pytest.raises(ValueError):
            CorpusFile(bad)

    def test_sample_count(self, tokenizer: Tokenizer, tmp_path: Path) -> None:
        """The loaded file reports the number of built samples."""

        out = self.build(tokenizer, tmp_path, ["one", "two", "three"])
        assert CorpusFile(out, preload=2).samples == 3

    def test_samples_roundtrip_to_original_tokens(
        self, tokenizer: Tokenizer, tmp_path: Path
    ) -> None:
        """Every sampled tensor matches the tokenization of one of the inputs."""

        texts = ["alpha beta", "gamma delta epsilon", "zeta"]
        out = self.build(tokenizer, tmp_path, texts)
        expected = {tuple(tokenizer.encode(wrap(t))) for t in texts}

        corpus = CorpusFile(out, preload=2)
        for sample in itertools.islice(iter(corpus), 20):
            assert tuple(sample) in expected

    def test_samples_are_int64_tensors(self, tokenizer: Tokenizer, tmp_path: Path) -> None:
        """Samples are 1-D int64 tensors starting with BOS and ending with EOS."""

        out = self.build(tokenizer, tmp_path, ["hello there"])
        sample = next(iter(CorpusFile(out, preload=1)))

        assert isinstance(sample, list)
        assert all(isinstance(x, int) for x in sample)
        assert len(sample) > 0
        assert sample[0] == tokenizer.bos_token
        assert sample[-1] == tokenizer.eos_token

    def test_sampling_covers_all_samples(self, tokenizer: Tokenizer, tmp_path: Path) -> None:
        """Random sampling eventually yields every sample in the corpus."""

        texts = ["red", "green", "blue"]
        out = self.build(tokenizer, tmp_path, texts)
        expected = {tuple(tokenizer.encode(wrap(t))) for t in texts}

        seen = {tuple(s) for s in itertools.islice(iter(CorpusFile(out, preload=2)), 300)}
        assert seen == expected
