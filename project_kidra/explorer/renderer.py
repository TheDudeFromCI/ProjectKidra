from project_kidra.explorer.similarity import SimilarityIndex


class ResultRenderer:
    """Builds the text lines shown for a query."""

    def __init__(self, index: SimilarityIndex, k: int = 10) -> None:
        """Show up to `k` results per query."""

        self._index = index
        self._k = k

    def render(self, query: str) -> list[str]:
        """Return result lines for `query`, lowercased to match the vocabulary."""

        query = query.strip().lower()
        if not query:
            return ["Type a token to see its nearest neighbours."]
        if self._index.has_token(query):
            return [
                f"{similarity * 100:6.1f}%  {token!r}"
                for token, similarity in self._index.nearest(query, self._k)
            ]
        lines = [f"{query!r} is not a token. Did you mean:"]
        lines += [f"  {token!r}" for token in self._index.suggest(query, self._k)]
        return lines
