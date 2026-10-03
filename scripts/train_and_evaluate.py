"""
End-to-end pipeline: load -> clean -> split -> train ALL recommenders ->
evaluate under the SAME protocol -> save artifacts + results.

Dataset: ml-32m only. Get it first with:
    python scripts/download_dataset.py

Run:
    python scripts/train_and_evaluate.py

The neural model benefits enormously from a GPU at this scale (32M
interactions) -- this is exactly what notebooks/00_colab_train_ml32m.ipynb
is for. This script runs fine on CPU too, just much slower for the neural
step; popularity / content-based / collaborative are all CPU-only anyway
and stay fast because they're built on sparse matrices, not dense ones.

All fitted models are saved to artifacts/models/ml-32m/ so the Streamlit
app (app/state.py) can load them instantly instead of retraining on every
first run.
"""
import argparse
import json
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
from src.coldstart.strategies import ColdStartManager
from src.evaluation.metrics import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                         help="Where the neural model writes latest.pt/best.pt while training. "
                              "On Colab, point this at a Google-Drive-mounted path (e.g. "
                              "/content/drive/MyDrive/CineSignal/checkpoints/neural_ncf) so a "
                              "runtime disconnect doesn't lose progress. Defaults to "
                              "artifacts/models/ml-32m/checkpoints/neural_ncf (local, NOT "
                              "Drive-persisted -- fine for local runs, risky on Colab).")
    parser.add_argument("--resume", action="store_true",
                         help="Resume neural training from <checkpoint-dir>/latest.pt if it exists.")
    parser.add_argument("--save-every-steps", type=int, default=None,
                         help="Force a mid-epoch checkpoint every N training steps (batches), "
                              "in addition to the automatic end-of-epoch save. Default: chosen "
                              "automatically from the dataset size. Lower this (e.g. 200-300) if "
                              "your Colab session tends to disconnect early/often.")
    args = parser.parse_args()

    t0 = time.time()
    print(f"=== Dataset: {config.DATASET} ===")
    if not config.RATINGS_CSV.exists():
        raise SystemExit(
            f"{config.RATINGS_CSV} not found.\n"
            f"Run `python scripts/download_dataset.py` first, or place the "
            f"extracted ml-32m archive at {config.RAW_DATASET_DIR}/"
        )

    ratings = clean_pipeline(loader.load_ratings())
    movies = loader.load_movies()
    tags = loader.load_tags()
    print(f"ratings={len(ratings):,}  movies={len(movies):,}  tags={len(tags):,}  ({time.time()-t0:.1f}s)")

    train, val, test = train_val_test_split(ratings)
    print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}")

    out_dir = config.MODEL_DATASET_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- fit every recommender on the SAME train split -------------------
    pop = PopularityRecommender().fit(train)
    pop.save(out_dir / "popularity.joblib")
    print(f"[popularity] fitted + saved, catalog eligible={len(pop.ranking_)} ({time.time()-t0:.1f}s)")

    cb = ContentBasedRecommender().fit(movies, tags)
    cb.save(out_dir / "content_based.joblib")
    print(f"[content]    fitted + saved, TF-IDF shape={cb.matrix_.shape} ({time.time()-t0:.1f}s)")

    cf = CollaborativeRecommender().fit(train)
    cf.save(out_dir / "collaborative.joblib")
    print(f"[collab]     fitted + saved, factors={cf.user_factors_.shape} ({time.time()-t0:.1f}s)")

    hybrid = HybridRecommender(cf, cb, pop)
    coldstart = ColdStartManager(cb, pop)

    neural = NeuralRecommender()
    neural_ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (out_dir / "checkpoints" / "neural_ncf")
    resume_from = (neural_ckpt_dir / "latest.pt") if args.resume else None
    neural.fit(train, val, verbose=True, checkpoint_dir=neural_ckpt_dir, resume_from=resume_from,
               save_every_steps=args.save_every_steps)
    print(f"[neural]     fitted + saved ({time.time()-t0:.1f}s)")
    neural.save(out_dir / "neural_ncf.pt")  # final inference-only artifact for the app

    # ---- evaluate all 5 under the same protocol ---------------------------
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

    eval_users_cap = 300  # keep 32M-scale evaluation tractable

    results = {}
    for name, fn in [("popularity", pop_fn), ("content_based", cb_profile_fn),
                      ("collaborative", cf_fn), ("hybrid", hybrid_fn), ("neural", neural_fn)]:
        r = evaluate_model(fn, test, k_values=config.EVAL_K_VALUES, max_users=eval_users_cap)
        results[name] = r
        print(f"[eval] {name:14s} NDCG@10={r['ndcg'][10]:.4f}  Precision@10={r['precision'][10]:.4f}  "
              f"Recall@10={r['recall'][10]:.4f}  HitRate@10={r['hit_rate'][10]:.4f}")

    out_path = config.PROJECT_ROOT / "docs" / f"evaluation_results_{config.DATASET}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved evaluation results -> {out_path}")
    print(f"Saved model artifacts -> {out_dir}/")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
