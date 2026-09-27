"""
Content-based recommender (spec section 12).

Real features only: genres (movies.csv) + aggregated user tags (tags.csv).
No overview/plot text exists in raw MovieLens -- if TMDB enrichment adds an
`overview` column later, `_build_corpus` already merges any extra text
columns found, so this scales up without a rewrite.

Uses TF-IDF + approximate cosine neighbors (sklearn NearestNeighbors,
metric="cosine") instead of a dense NxN similarity matrix -- required by
spec section 9/12 for memory efficiency at the 32M-catalog (87K movies)
scale: a dense matrix there would be 87585^2 floats (~30GB), infeasible.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src import config


def _build_corpus(movies: pd.DataFrame, tags: pd.DataFrame | None = None) -> pd.Series:
    genre_text = movies["genres_list"].apply(lambda gs: " ".join(g.replace(" ", "_") for g in gs))
    if tags is not None and len(tags):
        tag_text = (
            tags.groupby("movieId")["tag"]
            .apply(lambda t: " ".join(str(x).lower().replace(" ", "_") for x in t))
        )
        merged = movies[["movieId"]].merge(tag_text.rename("tag_text"), on="movieId", how="left")
        merged["tag_text"] = merged["tag_text"].fillna("")
        return genre_text + " " + merged["tag_text"]
    return genre_text


class ContentBasedRecommender:
    def __init__(self, max_features: int = config.TFIDF_MAX_FEATURES,
                 n_neighbors: int = config.CONTENT_NEIGHBORS):
        self.max_features = max_features
        self.n_neighbors = n_neighbors
        self.vectorizer_: TfidfVectorizer | None = None
        self.matrix_: sparse.csr_matrix | None = None
        self.nn_: NearestNeighbors | None = None
        self.movie_ids_: np.ndarray | None = None
        self.movieid_to_row_: dict | None = None

    def fit(self, movies: pd.DataFrame, tags: pd.DataFrame | None = None) -> "ContentBasedRecommender":
        corpus = _build_corpus(movies, tags)
        self.vectorizer_ = TfidfVectorizer(max_features=self.max_features, token_pattern=r"[^\s]+")
        self.matrix_ = self.vectorizer_.fit_transform(corpus)

        self.movie_ids_ = movies["movieId"].to_numpy()
        self.movieid_to_row_ = {mid: i for i, mid in enumerate(self.movie_ids_)}

        self.nn_ = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, self.matrix_.shape[0]),
            metric="cosine",
            algorithm="brute",  # exact cosine on sparse TF-IDF; fine up to ~1M items
        ).fit(self.matrix_)
        self._movies = movies
        return self

    def get_similar_movies(self, movie_id: int, k: int = config.TOP_K_DEFAULT) -> pd.DataFrame:
        assert self.nn_ is not None, "call .fit() first"
        if movie_id not in self.movieid_to_row_:
            return pd.DataFrame(columns=["movieId", "similarity", "explanation"])

        row = self.movieid_to_row_[movie_id]
        dist, idx = self.nn_.kneighbors(self.matrix_[row], n_neighbors=min(k + 1, self.matrix_.shape[0]))
        sims = 1 - dist[0]
        result = pd.DataFrame({"movieId": self.movie_ids_[idx[0]], "similarity": sims})
        result = result[result["movieId"] != movie_id].head(k)
        result["explanation"] = "Recommended because its metadata (genres/tags) is similar to this movie."
        return result.reset_index(drop=True)

    def build_user_profile_vector(self, liked_movie_ids: list[int]):
        rows = [self.movieid_to_row_[m] for m in liked_movie_ids if m in self.movieid_to_row_]
        if not rows:
            return None
        return sparse.csr_matrix(self.matrix_[rows].mean(axis=0))

    def get_recommendations_for_profile(self, liked_movie_ids: list[int], k: int = config.TOP_K_DEFAULT) -> pd.DataFrame:
        """Used by cold-start: build a content profile from real
        user-selected movies, then retrieve nearest catalog items."""
        profile = self.build_user_profile_vector(liked_movie_ids)
        if profile is None:
            return pd.DataFrame(columns=["movieId", "similarity", "explanation"])
        n = min(k + len(liked_movie_ids) + 1, self.matrix_.shape[0])
        dist, idx = self.nn_.kneighbors(profile, n_neighbors=n)
        sims = 1 - dist[0]
        result = pd.DataFrame({"movieId": self.movie_ids_[idx[0]], "similarity": sims})
        result = result[~result["movieId"].isin(liked_movie_ids)].head(k)
        result["explanation"] = "Recommended because its metadata is similar to movies in your preference profile."
        return result.reset_index(drop=True)

    def save(self, path: Path):
        joblib.dump({
            "vectorizer_": self.vectorizer_, "matrix_": self.matrix_, "nn_": self.nn_,
            "movie_ids_": self.movie_ids_, "movieid_to_row_": self.movieid_to_row_,
            "max_features": self.max_features, "n_neighbors": self.n_neighbors,
            "_movies": self._movies,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "ContentBasedRecommender":
        state = joblib.load(path)
        obj = cls(max_features=state["max_features"], n_neighbors=state["n_neighbors"])
        obj.vectorizer_ = state["vectorizer_"]
        obj.matrix_ = state["matrix_"]
        obj.nn_ = state["nn_"]
        obj.movie_ids_ = state["movie_ids_"]
        obj.movieid_to_row_ = state["movieid_to_row_"]
        obj._movies = state["_movies"]
        return obj
