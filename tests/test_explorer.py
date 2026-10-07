import torch

from project_kidra.explorer import ResultRenderer, SimilarityIndex

TOKENS = ["cat", "cats", "dog", "car"]
EMBEDDINGS = torch.tensor(
    [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]],
)


def make_index() -> SimilarityIndex:
    """Build a small index with known geometry."""

    return SimilarityIndex(TOKENS, EMBEDDINGS)


def test_nearest_orders_by_cosine_and_excludes_self() -> None:
    """Closest vector comes first and the query itself is omitted."""

    result = make_index().nearest("cat", k=3)
    assert [t for t, _ in result] == ["cats", "dog", "car"]
    assert result[0][1] > 0.99
    assert result[2][1] < 0


def test_suggest_prefers_prefix_matches() -> None:
    """Unknown text yields prefix matches before fuzzy ones."""

    assert make_index().suggest("ca", k=3) == ["cat", "car", "cats"]


def test_suggest_falls_back_to_fuzzy() -> None:
    """Typos still produce close tokens."""

    assert make_index().suggest("dgo", k=2)[0] == "dog"


def test_renderer_shows_percent_for_tokens() -> None:
    """Known tokens render percentage lines."""

    lines = ResultRenderer(make_index(), k=2).render("CAT")
    assert len(lines) == 2
    assert "%" in lines[0] and "'cats'" in lines[0]


def test_renderer_suggests_for_unknown_text() -> None:
    """Unknown text renders a suggestion list."""

    lines = ResultRenderer(make_index(), k=2).render("catz")
    assert "not a token" in lines[0]
    assert len(lines) == 3
