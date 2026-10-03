"""
Re-run evaluation only, on already-trained artifacts -- no retraining.

Why this exists: the first train_and_evaluate.py run evaluated only the
FIRST 300 users (lowest userId = oldest/sparsest MovieLens accounts), not a
random sample. That's enough rows for popularity/CF/hybrid to show signal,
but for neural (which needs more headroom to show ranking quality) it can
land on literal 0.0000 by chance -- this reruns eval on a larger, RANDOM
user sample using the artifacts already saved to disk, to check whether the
neural=0.0000 was noise or real.

Run:
    python scripts/reevaluate.py --n-users 2000
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data import loader
from src.data.cleaner import clean_pipeline
from src.data.splitter import train_val_test_split
from src.models.popularity import PopularityRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender
from src.models.hybrid import HybridRecommender
from src.models.neural import NeuralRecommender
from src.evaluation.metrics import evaluate_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=2000,
                     help="How many RANDOM test users to evaluate on (default 2000, up from the original 300).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = config.MODEL_DATASET_DIR
    print(f"Loading artifacts from {out_dir} ...")
    pop = PopularityRecommender.load(out_dir / "popularity.joblib")
    cb = ContentBasedRecommender.load(out_dir / "content_based.joblib")
    cf = CollaborativeRecommender.load(out_dir / "collaborative.joblib")
    neural = NeuralRecommender.load(out_dir / "neural_ncf.pt")
    hybrid = HybridRecommender(cf, cb, pop)

    # neural.get_user_recommendations() needs self._seen (train interactions
    # per user) -- .load() doesn't restore it (it's not saved, see neural.py
    # docstring), so we rebuild train/test the same way train_and_evaluate.py
    # did, and re-attach it here.
    print("Rebuilding train/test split (same pipeline, same seed as training) ...")
    ratings = clean_pipeline(loader.load_ratings())
    train, val, test = train_val_test_split(ratings)
    neural._seen = train.groupby("userId")["movieId"].apply(set).to_dict()
    cf._seen = neural._seen  # cf/hybrid's cb_profile_fn also reads this

    def cf_fn(uid, k):
        return cf.get_user_recommendations(uid, k)["movieId"].tolist()

    def cb_profile_fn(uid, k):
        liked = list(cf._seen.get(uid, set()))[:20]
        if not liked:
            return []
        return cb.get_recommendations_for_profile(liked, k)["movieId"].tolist()

    def pop_fn(uid, k):
        return pop.get_user_recommendations(uid, k)["movieId"].tolist()

    def hybrid_fn(uid, k):
        return hybrid.get_user_recommendations(uid, k)["movieId"].tolist()

    def neural_fn(uid, k):
        return neural.get_user_recommendations(uid, k)["movieId"].tolist()

    # ---- random (not first-N) user sample, reproducible via --seed --------
    relevant_by_user = test.groupby("userId")["movieId"].apply(set).to_dict()
    all_users = list(relevant_by_user.keys())
    random.seed(args.seed)
    sample_users = random.sample(all_users, min(args.n_users, len(all_users)))
    print(f"Evaluating on {len(sample_users)} RANDOM users (seed={args.seed}), "
          f"vs original 300 lowest-userId users.")

    # evaluate_model takes the first N of whatever dict it's given, so build
    # a test_df restricted to exactly our random sample, in that order.
    test_sample = test[test["userId"].isin(sample_users)]

    results = {}
    for name, fn in [("popularity", pop_fn), ("content_based", cb_profile_fn),
                      ("collaborative", cf_fn), ("hybrid", hybrid_fn), ("neural", neural_fn)]:
        r = evaluate_model(fn, test_sample, k_values=config.EVAL_K_VALUES, max_users=len(sample_users))
        results[name] = r
        print(f"[eval] {name:14s} NDCG@10={r['ndcg'][10]:.4f}  Precision@10={r['precision'][10]:.4f}  "
              f"Recall@10={r['recall'][10]:.4f}  HitRate@10={r['hit_rate'][10]:.4f}")

    out_path = config.PROJECT_ROOT / "docs" / f"evaluation_results_{config.DATASET}_resampled.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out_path}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()