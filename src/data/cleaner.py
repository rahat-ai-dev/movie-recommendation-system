"""Cleaning stage: dedupe, drop invalid rows, build stable ID mappings.

These are pure functions (no I/O) so they're easy to unit test with tiny
real-data slices, per the spec's "pure-function tests" preference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def dedupe_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest rating if a user rated the same movie twice."""
    return (
        ratings.sort_values("timestamp")
        .drop_duplicates(subset=["userId", "movieId"], keep="last")
        .reset_index(drop=True)
    )


def drop_invalid_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    valid = ratings["rating"].between(0.5, 5.0) & ratings["userId"].notna() & ratings["movieId"].notna()
    return ratings.loc[valid].reset_index(drop=True)


def build_id_mappings(ratings: pd.DataFrame) -> tuple[dict, dict, dict, dict]:
    """Map raw MovieLens userId/movieId (sparse, non-contiguous) to dense
    0..N-1 integer indices required for embedding layers / sparse matrices.
    """
    user_ids = np.sort(ratings["userId"].unique())
    movie_ids = np.sort(ratings["movieId"].unique())

    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    idx_to_user = {i: uid for uid, i in user_to_idx.items()}
    movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    idx_to_movie = {i: mid for mid, i in movie_to_idx.items()}

    return user_to_idx, idx_to_user, movie_to_idx, idx_to_movie


def clean_pipeline(ratings: pd.DataFrame) -> pd.DataFrame:
    return dedupe_ratings(drop_invalid_ratings(ratings))
