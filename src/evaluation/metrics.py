"""
Top-K ranking metrics (spec section 31). All are pure functions of
(recommended_ids, relevant_ids) so they're trivially unit-testable.
"""
from __future__ import annotations

import numpy as np


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for m in rec_k if m in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for m in rec_k if m in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    rec_k = recommended[:k]
    return 1.0 if any(m in relevant for m in rec_k) else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    rec_k = recommended[:k]
    dcg = sum(1.0 / np.log2(i + 2) for i, m in enumerate(rec_k) if m in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    rec_k = recommended[:k]
    hits, sum_precisions = 0, 0.0
    for i, m in enumerate(rec_k, start=1):
        if m in relevant:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / min(len(relevant), k) if relevant else 0.0


def evaluate_model(get_recommendations_fn, test_df, k_values=(5, 10, 20), max_users: int | None = None):
    """
    get_recommendations_fn(user_id, k) -> list[movieId] (already ranked, seen-items excluded)
    test_df: held-out interactions with columns [userId, movieId]
    Returns a dict of metric -> {k: mean_score}
    """
    relevant_by_user = test_df.groupby("userId")["movieId"].apply(set).to_dict()
    users = list(relevant_by_user.keys())
    if max_users:
        users = users[:max_users]

    max_k = max(k_values)
    results = {m: {k: [] for k in k_values} for m in
               ("precision", "recall", "ndcg", "hit_rate", "map")}

    for uid in users:
        relevant = relevant_by_user[uid]
        recs = get_recommendations_fn(uid, max_k)
        for k in k_values:
            results["precision"][k].append(precision_at_k(recs, relevant, k))
            results["recall"][k].append(recall_at_k(recs, relevant, k))
            results["ndcg"][k].append(ndcg_at_k(recs, relevant, k))
            results["hit_rate"][k].append(hit_rate_at_k(recs, relevant, k))
            results["map"][k].append(average_precision_at_k(recs, relevant, k))

    return {
        metric: {k: float(np.mean(vals)) if vals else 0.0 for k, vals in per_k.items()}
        for metric, per_k in results.items()
    }
