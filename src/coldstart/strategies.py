"""
Cold-start handling (spec section 15). Every path documented and backed by
real data -- no fake ratings are ever synthesized for a new user.
"""
from __future__ import annotations

import pandas as pd

from src import config


class ColdStartManager:
    def __init__(self, content_based, popularity):
        self.cb = content_based
        self.pop = popularity

    def new_user_from_selections(self, liked_movie_ids: list[int], k: int = config.TOP_K_DEFAULT) -> pd.DataFrame:
        """Flow: selected real movies -> content profile -> similarity
        retrieval -> popularity-aware re-rank (spec section 15/26)."""
        if not liked_movie_ids:
            out = self.pop.recommend(k=k)
            out["explanation"] = "No preferences given yet -- showing popular picks."
            return out

        candidates = self.cb.get_recommendations_for_profile(liked_movie_ids, k=k * 3)
        if candidates.empty:
            out = self.pop.recommend(k=k)
            out["explanation"] = "Selections not in catalog -- popularity fallback."
            return out

        pop_lookup = self.pop.ranking_.set_index("movieId")["weighted_rating"] if self.pop.ranking_ is not None else pd.Series(dtype=float)
        candidates["pop_score"] = candidates["movieId"].map(pop_lookup).fillna(0.0)
        lo, hi = candidates["pop_score"].min(), candidates["pop_score"].max()
        candidates["pop_norm"] = 0.0 if hi - lo < 1e-9 else (candidates["pop_score"] - lo) / (hi - lo)
        candidates["final_score"] = 0.7 * candidates["similarity"] + 0.3 * candidates["pop_norm"]
        candidates = candidates.sort_values("final_score", ascending=False).head(k)
        candidates["explanation"] = "Built from the movies you picked (content similarity) + popularity re-ranking."
        return candidates.reset_index(drop=True)

    def new_user_from_genres(self, liked_genres: list[str], movies: pd.DataFrame, k: int = config.TOP_K_DEFAULT) -> pd.DataFrame:
        mask = movies["genres_list"].apply(lambda gs: any(g in gs for g in liked_genres))
        pool_ids = movies.loc[mask, "movieId"]
        if self.pop.ranking_ is None or pool_ids.empty:
            return self.pop.recommend(k=k)
        ranked = self.pop.ranking_[self.pop.ranking_["movieId"].isin(pool_ids)].head(k).copy()
        ranked["explanation"] = f"Popular movies in your selected genres: {', '.join(liked_genres)}."
        return ranked

    def unknown_user(self, k: int = config.TOP_K_DEFAULT) -> pd.DataFrame:
        out = self.pop.recommend(k=k)
        out["explanation"] = "Unknown user -- popularity-based cold-start."
        return out

    def new_item_similarity(self, movie_id: int, k: int = config.TOP_K_DEFAULT) -> pd.DataFrame:
        """New item with no collaborative history -- rely purely on content
        signals (this is exactly what ContentBasedRecommender already does;
        exposed here under the cold-start name for clarity of intent)."""
        return self.cb.get_similar_movies(movie_id, k=k)

    def sparse_user(self, user_id: int, hybrid, k: int = config.TOP_K_DEFAULT, alpha: float = 0.3) -> pd.DataFrame:
        """Sparse user: down-weight collaborative signal (noisy with few
        ratings) in favor of content, via a lower alpha than the default."""
        return hybrid.get_user_recommendations(user_id, k=k, alpha=alpha)
