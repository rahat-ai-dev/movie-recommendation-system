from src.evaluation.metrics import (
    precision_at_k, recall_at_k, hit_rate_at_k, ndcg_at_k, average_precision_at_k
)


def test_precision_at_k_basic():
    recommended = [1, 2, 3, 4, 5]
    relevant = {2, 4, 99}
    assert precision_at_k(recommended, relevant, 5) == 2 / 5


def test_recall_at_k_basic():
    recommended = [1, 2, 3]
    relevant = {2, 3, 4, 5}
    assert recall_at_k(recommended, relevant, 3) == 2 / 4


def test_hit_rate_at_k():
    assert hit_rate_at_k([1, 2, 3], {5, 6}, 3) == 0.0
    assert hit_rate_at_k([1, 2, 3], {3, 6}, 3) == 1.0


def test_ndcg_perfect_ranking_is_one():
    recommended = [10, 20, 30]
    relevant = {10, 20, 30}
    assert abs(ndcg_at_k(recommended, relevant, 3) - 1.0) < 1e-9


def test_ndcg_worse_ranking_is_lower():
    relevant = {30}
    best = ndcg_at_k([30, 10, 20], relevant, 3)
    worst = ndcg_at_k([10, 20, 30], relevant, 3)
    assert best > worst


def test_average_precision_at_k():
    recommended = [1, 2, 3, 4]
    relevant = {1, 3}
    # hits at rank1 (p=1/1) and rank3 (p=2/3) -> avg = (1 + 2/3)/2
    assert abs(average_precision_at_k(recommended, relevant, 4) - (1 + 2 / 3) / 2) < 1e-9


def test_empty_relevant_returns_zero():
    assert recall_at_k([1, 2, 3], set(), 3) == 0.0
    assert average_precision_at_k([1, 2, 3], set(), 3) == 0.0
