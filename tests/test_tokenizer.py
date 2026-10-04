from project_kidra import tokenizer


def test_tokenize_text() -> None:
    t = tokenizer.Tokenizer()
    
    # Set the tokens manually for testing purposes
    t.tokens = ["this", "is", "a", "test", ".", "another", "blue", "numbers", "1", "2", "3", "and", "symbols", "#", "$", "%", "!"]

    tokens = t.tokenize("This is a test.")
    assert tokens == [0, 1, 2, 3, 4]

    tokens = t.tokenize("Another test! Blue!")
    assert tokens == [5, 3, 16, 6, 16]

    tokens = t.tokenize("Numbers 123 and symbols #$%")
    assert tokens == [7, 8, 9, 10, 11, 12, 13, 14, 15]

    tokens = t.tokenize("")
    assert tokens == []


def test_tokens_file_loading() -> None:
    t = tokenizer.Tokenizer()

    assert t.vocab_size() > 0
    assert isinstance(t.tokens, list)
    assert all(isinstance(token, str) for token in t.tokens)

def test_special_tokens_round_trip() -> None:
    """Special tokens follow the vocabulary, are recognised verbatim, and decode back."""

    t = tokenizer.Tokenizer()
    t.tokens = ["call", "search"]
    ids = t.tokenize("<|tool_begin|> call search <|tool_end|>")
    assert ids == [t.special_id(tokenizer.TOOL_BEGIN_TOKEN), 0, 1, t.special_id(tokenizer.TOOL_END_TOKEN)]
    assert t.decode(ids) == "<|tool_begin|> call search <|tool_end|>"
    assert t.vocab_size() == 2 + len(tokenizer.SPECIAL_TOKENS)


def test_unknown_words_dropped_or_mapped_to_unk() -> None:
    """Unknown words are dropped by default and become <|unk|> when requested."""

    t = tokenizer.Tokenizer()
    t.tokens = ["known"]
    assert t.tokenize("known mystery") == [0]
    assert t.tokenize("known mystery", keep_unknown=True) == [0, t.special_id(tokenizer.UNK_TOKEN)]
