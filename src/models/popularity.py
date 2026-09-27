"""
Popularity baseline -- spec section 10 explicitly forbids "sort by raw
average rating" (a movie with one 5-star rating would beat a movie with
10,000 ratings averaging 4.5). We use the IMDB-style Bayesian weighted
rating instead:

    WR = (v / (v + m)) * R + (m / (v + m)) * C

    v = number of ratings for the movie
    m = minimum ratings threshold to be considered (config.POPULARITY_CONFIDENCE_M)
    R = movie's own mean rating
    C = mean rating across the whole (eligible) catalog
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src import config


class PopularityRecommender:
    def __init__(self, min_ratings: int = config.POPULARITY_MIN_RATINGS,
                 m: int = config.POPULARITY_CONFIDENCE_M):
        self.min_ratings = min_ratings
        self.m = m
        self.ranking_: pd.DataFrame | None = None
        self._seen: dict[int, set] | None = None

    def fit(self, ratings: pd.DataFrame) -> "PopularityRecommender":
        stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"]).reset_index()
        stats.columns = ["movieId", "num_ratings", "avg_rating"]
        stats = stats[stats["num_ratings"] >= self.min_ratings].copy()

        C = stats["avg_rating"].mean() if len(stats) else ratings["rating"].mean()
        v, R, m = stats["num_ratings"], stats["avg_rating"], self.m
        stats["weighted_rating"] = (v / (v + m)) * R + (m / (v + m)) * C
        self.global_mean_ = C

        self.ranking_ = stats.sort_values("weighted_rating", ascending=False).reset_index(drop=True)
        self._seen = ratings.groupby("userId")["movieId"].apply(set).to_dict()
        return self

    def recommend(self, k: int = config.TOP_K_DEFAULT, exclude_seen_for_user: int | None = None) -> pd.DataFrame:
        assert self.ranking_ is not None, "call .fit() first"
        pool = self.ranking_
        if exclude_seen_for_user is not None and self._seen:
            seen = self._seen.get(exclude_seen_for_user, set())
            pool = pool[~pool["movieId"].isin(seen)]
        out = pool.head(k).copy()
        out["explanation"] = (
            "Popular pick: high volume of ratings "
            "(v/(v+m) confidence weighting, m="
            + str(self.m) + ") with a strong average score."
        )
        return out[["movieId", "num_ratings", "avg_rating", "weighted_rating", "explanation"]]

    def get_user_recommendations(self, user_id: int, k: int = config.TOP_K_DEFAULT) -> pd.DataFrame:
        """Same signature style as the other recommenders for the hybrid
        layer / app to call polymorphically."""
        return self.recommend(k=k, exclude_seen_for_user=user_id)

    def save(self, path: Path):
        joblib.dump({
            "ranking_": self.ranking_, "_seen": self._seen,
            "global_mean_": self.global_mean_,
            "min_ratings": self.min_ratings, "m": self.m,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "PopularityRecommender":
        state = joblib.load(path)
        obj = cls(min_ratings=state["min_ratings"], m=state["m"])
        obj.ranking_ = state["ranking_"]
        obj._seen = state["_seen"]
        obj.global_mean_ = state["global_mean_"]
        return obj
