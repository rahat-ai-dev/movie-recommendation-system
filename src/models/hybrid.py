"""
Hybrid recommender (spec section 13).

hybrid_score = alpha * collaborative_score_norm + (1-alpha) * content_score_norm

Both component scores are min-max normalized per-request before blending
(they live on different scales: CF scores are ~[0.5, 5], content is
cosine similarity in [0, 1]) so alpha behaves predictably.

Falls back automatically:
  - unknown/new user with no CF vector -> content/popularity path (cold-start)
  - user with < MIN interactions          -> blend still runs, CF contributes
    a noisier but present signal (matrix factorization does produce a row
    for any user seen during .fit(), however sparse).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def _minmax(s: pd.Series) -> pd.Series:
    if s.empty:
        return s
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(np.ones(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


class HybridRecommender:
    def __init__(self, collaborative, content_based, popularity,
                 alpha: float = config.HYBRID_DEFAULT_ALPHA):
        self.cf = collaborative
        self.cb = content_based
        self.pop = popularity
        self.alpha = alpha

    def get_user_recommendations(self, user_id: int, k: int = config.TOP_K_DEFAULT,
                                  alpha: float | None = None) -> pd.DataFrame:
        alpha = self.alpha if alpha is None else alpha

        if user_id not in self.cf.user_to_idx:
            # New/unknown user to the CF model -> cold-start fallback (popularity).
            out = self.pop.get_user_recommendations(user_id, k=k)
            out["explanation"] = "New/unknown user: popularity-based cold-start fallback."
            return out

        cf_pool = self.cf.get_user_recommendations(user_id, k=max(k * 5, 50))
        if cf_pool.empty:
            return self.pop.get_user_recommendations(user_id, k=k)

        candidate_ids = cf_pool["movieId"].tolist()
        liked = list(self.cf._seen.get(user_id, set()))[:20]  # recent-ish liked items as content anchor
        cb_scores = {}
        if liked:
            cb_pool = self.cb.get_recommendations_for_profile(liked, k=len(candidate_ids) * 2)
            cb_scores = dict(zip(cb_pool["movieId"], cb_pool["similarity"]))

        cf_pool["cf_norm"] = _minmax(cf_pool["cf_score"])
        cf_pool["content_score"] = cf_pool["movieId"].map(cb_scores).fillna(0.0)
        cf_pool["content_norm"] = _minmax(cf_pool["content_score"])
        cf_pool["hybrid_score"] = alpha * cf_pool["cf_norm"] + (1 - alpha) * cf_pool["content_norm"]
        cf_pool["explanation"] = (
            f"Hybrid: {alpha:.2f}\u00d7collaborative + {1-alpha:.2f}\u00d7content signal."
        )

        out = cf_pool.sort_values("hybrid_score", ascending=False).head(k)
        return out[["movieId", "hybrid_score", "cf_score", "content_score", "explanation"]].reset_index(drop=True)
