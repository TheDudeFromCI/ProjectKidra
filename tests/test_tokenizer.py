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