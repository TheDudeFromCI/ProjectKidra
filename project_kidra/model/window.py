from __future__ import annotations

from project_kidra.model.cache import KVCache


class ContextWindow:
    """Context window for managing local key-value caches and token sequences."""

    def __init__(
        self,
        index: int,
        id: int,
        n_layers: int,
        n_local_ctx: int,
        vocab_size: int,
        free_token: int,
    ):
        """Initialize the context window with its owner model, index, window ID, and active status."""

        self.index = index
        self.id = id
        self.tokens: list[int] = []
        self.pending: list[int] = []
        self.position = 0
        self.cache = [KVCache(n_local_ctx) for _ in range(n_layers)]
        self.n_local_ctx = n_local_ctx
        self.vocab_size = vocab_size
        self.free_token = free_token

    def append(self, token: int) -> ContextWindow:
        """Append a new token to the context window."""

        if token < 0 or token >= self.vocab_size:
            raise ValueError("Token ID outside vocabulary")

        self.tokens.append(token)
        self.tokens = self.tokens[-self.n_local_ctx :]
        return self

    def expect(self, tokens: list[int]) -> ContextWindow:
        """Expect certain tokens to appear next in the context window, updating the pending list accordingly."""

        if any(v < 0 or v >= self.vocab_size for v in tokens):
            raise ValueError("Token ID outside vocabulary")

        self.pending = tokens
        return self

    def pop_pending(self) -> int:
        """Pop the next expected token from the pending list, or return the free token if none are pending."""

        if self.pending:
            return self.pending.pop(0)

        return self.free_token

    def detach_cache(self) -> None:
        """Detach all key-value caches."""

        for cache in self.cache:
            cache.detach()
