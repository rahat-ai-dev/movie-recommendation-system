"""
Collaborative filtering (spec section 11) via matrix factorization on a
SPARSE user-item matrix -- never a dense matrix (610 users x 9724 movies is
fine dense, but 200K+ users x 87K movies at ml-32m scale is not: that would
be ~140GB dense vs. ~a few hundred MB sparse).

Method: mean-centered ratings -> scipy.sparse.linalg.svds (truncated SVD)
-> user_factors (U*sqrt(S)), item_factors (V*sqrt(S)). Scoring is a dot
product of latent vectors, ranked, with already-seen items excluded.

Documented limitations (spec section 11):
  - Pure matrix factorization has no cold-start handling for new users/items
    (that's why ColdStartManager exists separately).
  - SVD on explicit ratings treats missing entries as "centered zero" rather
    than "unknown" -- a well-known approximation of true matrix completion;
    acceptable for a portfolio-scale demo, called out here rather than
    hidden.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

from src import config
from src.data.cleaner import build_id_mappings


class CollaborativeRecommender:
    def __init__(self, n_factors: int = config.CF_LATENT_FACTORS,
                 random_state: int = config.CF_RANDOM_STATE):
        self.n_factors = n_factors
        self.random_state = random_state

    def fit(self, ratings: pd.DataFrame) -> "CollaborativeRecommender":
        self.user_to_idx, self.idx_to_user, self.movie_to_idx, self.idx_to_movie = build_id_mappings(ratings)
        n_users, n_items = len(self.user_to_idx), len(self.movie_to_idx)

        rows = ratings["userId"].map(self.user_to_idx).to_numpy()
        cols = ratings["movieId"].map(self.movie_to_idx).to_numpy()
        vals = ratings["rating"].to_numpy(dtype=np.float32)

        self.user_means_ = np.zeros(n_users, dtype=np.float32)
        sums = np.zeros(n_users, dtype=np.float64)
        counts = np.zeros(n_users, dtype=np.int64)
        np.add.at(sums, rows, vals)
        np.add.at(counts, rows, 1)
        self.user_means_ = (sums / np.maximum(counts, 1)).astype(np.float32)

        centered_vals = vals - self.user_means_[rows]
        R = sparse.csr_matrix((centered_vals, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)

        k = min(self.n_factors, min(R.shape) - 1)
        u, s, vt = svds(R, k=k, random_state=self.random_state)
        order = np.argsort(-s)
        s, u, vt = s[order], u[:, order], vt[order, :]

        sqrt_s = np.sqrt(s)
        self.user_factors_ = (u * sqrt_s).astype(np.float32)      # (n_users, k)
        self.item_factors_ = (vt.T * sqrt_s).astype(np.float32)   # (n_items, k)

        self._seen = ratings.groupby("userId")["movieId"].apply(set).to_dict()
        self.n_users_, self.n_items_ = n_users, n_items
        return self

    def predict_score(self, user_id: int, movie_id: int) -> float | None:
        if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx:
            return None
        u = self.user_to_idx[user_id]
        i = self.movie_to_idx[movie_id]
        return float(self.user_means_[u] + self.user_factors_[u] @ self.item_factors_[i])

    def get_user_recommendations(self, user_id: int, k: int = config.TOP_K_DEFAULT) -> pd.DataFrame:
        if user_id not in self.user_to_idx:
            # unknown user -> caller should fall back to cold-start / popularity
            return pd.DataFrame(columns=["movieId", "cf_score", "explanation"])

        u = self.user_to_idx[user_id]
        scores = self.user_means_[u] + self.item_factors_ @ self.user_factors_[u]
        seen = self._seen.get(user_id, set())
        seen_idx = {self.movie_to_idx[m] for m in seen if m in self.movie_to_idx}

        order = np.argsort(-scores)
        results = []
        for idx in order:
            if idx in seen_idx:
                continue
            results.append((self.idx_to_movie[idx], float(scores[idx])))
            if len(results) >= k:
                break

        out = pd.DataFrame(results, columns=["movieId", "cf_score"])
        out["explanation"] = "Recommended because users with similar rating patterns also liked this movie."
        return out

    def save(self, path: Path):
        joblib.dump({
            "user_to_idx": self.user_to_idx, "idx_to_user": self.idx_to_user,
            "movie_to_idx": self.movie_to_idx, "idx_to_movie": self.idx_to_movie,
            "user_means_": self.user_means_, "user_factors_": self.user_factors_,
            "item_factors_": self.item_factors_, "_seen": self._seen,
            "n_users_": self.n_users_, "n_items_": self.n_items_,
            "n_factors": self.n_factors, "random_state": self.random_state,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "CollaborativeRecommender":
        state = joblib.load(path)
        obj = cls(n_factors=state["n_factors"], random_state=state["random_state"])
        for key in ("user_to_idx", "idx_to_user", "movie_to_idx", "idx_to_movie",
                    "user_means_", "user_factors_", "item_factors_", "_seen",
                    "n_users_", "n_items_"):
            setattr(obj, key, state[key])
        return obj
