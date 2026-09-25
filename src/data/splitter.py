"""
Time-aware, leakage-free split.

Per spec section 8: earlier interactions -> train, later -> val/test,
split PER USER along their own timeline (a global time cut would starve
early-joining users of any test data and is a common mistake).

Users below MIN_INTERACTIONS_FOR_TEST_USER are excluded from val/test
(their few ratings stay in train only) -- can't meaningfully evaluate
Top-K ranking with 1-2 held-out items.
"""
from __future__ import annotations

import pandas as pd

from src import config


def time_aware_split(
    ratings: pd.DataFrame,
    test_fraction: float = config.TIME_SPLIT_TEST_FRACTION,
    min_interactions: int = config.MIN_INTERACTIONS_FOR_TEST_USER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = ratings.sort_values(["userId", "timestamp"])
    counts = ratings.groupby("userId")["movieId"].transform("count")
    eligible = counts >= min_interactions

    rank = ratings.groupby("userId").cumcount()
    total = ratings.groupby("userId")["movieId"].transform("count")
    cutoff = (total * (1 - test_fraction)).astype(int)

    is_test = eligible & (rank >= cutoff)
    train = ratings.loc[~is_test].reset_index(drop=True)
    test = ratings.loc[is_test].reset_index(drop=True)
    return train, test


def train_val_test_split(
    ratings: pd.DataFrame,
    val_fraction: float = 0.1,
    test_fraction: float = config.TIME_SPLIT_TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train / val / test, all time-ordered per user. val is carved out of
    what would otherwise be train, using the same time-respecting logic."""
    train_full, test = time_aware_split(ratings, test_fraction=test_fraction)
    train, val = time_aware_split(train_full, test_fraction=val_fraction)
    return train, val, test
