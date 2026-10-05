from project_kidra.corpus import (
    Tokenizer,
    TOOL_BEGIN_TOKEN,
    TOOL_END_TOKEN,
    UNK_TOKEN,
    SPECIAL_TOKENS,
    NAME_BEGIN,
    LETTER_TOKENS,
    NEW_LINE,
)


def test_tokenize_text() -> None:
    t = Tokenizer(
        [
            "this",
            "is",
            "the",
            "test",
            ".",
            "another",
            "blue",
            "numbers",
            "1",
            "2",
            "3",
            "and",
            "symbols",
            "#",
            "$",
            "%",
            "!",
        ]
    )

    tokens = t.tokenize("This is the test.")
    assert tokens == [0, 1, 2, 3, 4]

    tokens = t.tokenize("Another test! Blue!")
    assert tokens == [5, 3, 16, 6, 16]

    tokens = t.tokenize("Numbers 123 and symbols #$%")
    assert tokens == [7, 8, 9, 10, 11, 12, 13, 14, 15]

    tokens = t.tokenize("")
    assert tokens == []


def test_tokens_file_loading() -> None:
    t = Tokenizer.default_vocab()

    assert t.vocab_size() > 0
    assert isinstance(t._tokens, list)
    assert all(isinstance(token, str) for token in t._tokens)


def test_special_tokens_round_trip() -> None:
    """Special tokens follow the vocabulary, are recognised verbatim, and decode back."""

    t = Tokenizer(["call", "search"])
    ids = t.tokenize("<|tool_begin|>call search<|tool_end|>")

    assert ids == [
        t.token_id(TOOL_BEGIN_TOKEN),
        t.token_id("call"),
        t.token_id("search"),
        t.token_id(TOOL_END_TOKEN),
    ]

    assert t.decode(ids) == "<|tool_begin|>call search<|tool_end|>"
    assert t.vocab_size() == 2 + len(SPECIAL_TOKENS) + len(LETTER_TOKENS)


def test_unknown_words_dropped_or_mapped_to_unk() -> None:
    """Unknown words are dropped by default and become <|unk|> when requested."""

    t = Tokenizer(["known"])
    assert t.tokenize("known!") == [0]
    assert t.tokenize("known!", keep_unknown=True) == [
        0,
        t.token_id(UNK_TOKEN),
    ]


def test_proper_names() -> None:
    """Proper names are tokenized correctly and consistently."""

    t = Tokenizer(["hi", "!"])
    tokens = t.tokenize("Hi Alice!")

    hi_id = t.token_id("hi")
    name_id = t.token_id(NAME_BEGIN)
    a_id = t.token_id("a")
    l_id = t.token_id("l")
    i_id = t.token_id("i")
    c_id = t.token_id("c")
    e_id = t.token_id("e")
    exclamation_id = t.token_id("!")

    assert tokens == [hi_id, name_id, a_id, l_id, i_id, c_id, e_id, exclamation_id]


def test_newlines() -> None:
    """Newlines are tokenized correctly."""

    t = Tokenizer(["line", "break"])
    tokens = t.tokenize("line\nbreak")
    assert tokens == [0, t.token_id(NEW_LINE), 1]
