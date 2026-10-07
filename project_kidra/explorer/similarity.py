import difflib

import torch


class SimilarityIndex:
    """Cosine-similarity lookup over token embeddings with string-distance fallback."""

    def __init__(self, tokens: list[str], embeddings: torch.Tensor) -> None:
        """Index `embeddings` (vocab, dim), where row i belongs to `tokens[i]`."""

        if len(tokens) != embeddings.shape[0]:
            raise ValueError("token count does not match embedding rows")
        self._tokens = tokens
        self._ids = {token: i for i, token in enumerate(tokens)}
        self._normalized = torch.nn.functional.normalize(embeddings.float(), dim=1)

    def has_token(self, text: str) -> bool:
        """Return True if `text` is exactly a known token."""

        return text in self._ids

    def nearest(self, token: str, k: int = 10) -> list[tuple[str, float]]:
        """Return the `k` most similar other tokens as (token, cosine similarity)."""

        index = self._ids[token]
        scores = self._normalized @ self._normalized[index]
        scores[index] = float("-inf")
        values, ids = torch.topk(scores, min(k, len(self._tokens) - 1))
        return [(self._tokens[i], v) for i, v in zip(ids.tolist(), values.tolist())]

    def suggest(self, text: str, k: int = 10) -> list[str]:
        """Return up to `k` tokens closest to `text` by string distance, prefix matches first."""

        prefixed = [t for t in self._tokens if t.startswith(text)]
        prefixed.sort(key=len)
        result = prefixed[:k]
        if len(result) < k:
            close = difflib.get_close_matches(
                text, self._tokens, n=k + len(result), cutoff=0.0
            )
            result += [t for t in close if t not in result][: k - len(result)]
        return result
