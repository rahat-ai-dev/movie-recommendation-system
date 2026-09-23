# CineSignal — Movie Recommendation System

A portfolio-grade recommendation platform built on **real MovieLens data** (GroupLens
Research) — no synthetic ratings, no fabricated users, no invented metadata anywhere
in the pipeline.

Five recommenders, one shared data pipeline, one evaluation protocol, one Streamlit app,
trained on the **full ml-32m dataset** (32,000,204 real ratings, 87,585 movies,
200,948 users):

| Recommender | Method | File |
|---|---|---|
| Popularity | Bayesian-weighted rating (not raw average) | `src/models/popularity.py` |
| Collaborative Filtering | Sparse truncated-SVD matrix factorization | `src/models/collaborative.py` |
| Content-Based | TF-IDF (genres+tags) + cosine kNN | `src/models/content_based.py` |
| Hybrid | Configurable alpha blend of CF + content | `src/models/hybrid.py` |
| Neural | PyTorch embedding-MLP NCF | `src/models/neural.py` |

## Quickstart

```bash
python -m venv venv && source venv/bin/activate     # optional but recommended
pip install -r requirements.txt

# 1. Get the data (official GroupLens source, ~240MB compressed / ~900MB extracted)
python scripts/download_dataset.py
#   -> extracts to data/raw/ml-32m/{ratings,movies,tags,links}.csv
#   (or pass --zip /path/to/ml-32m.zip if you already have the archive)

# 2. Inspect + validate the data (writes docs/dataset_summary.json)
python -m src.data.validator

# 3. Train + evaluate all 5 recommenders on the same split
#    (see "Training at 32M-row scale" below -- do this on a GPU via Colab
#    for the neural model; the other 4 are fast on CPU regardless)
python scripts/train_and_evaluate.py

# 4. Run the app -- it loads the artifacts saved in step 3 instantly
streamlit run app/streamlit_app.py
```

**TMDB posters/backdrops/overviews:** a `.env` file with a working `TMDB_API_KEY`
already ships with this project, so posters show up in the app UI out of the box.
Rotate the key any time by editing `.env` (see `.env.example` for the format). The
app **works fully without one** too — it degrades to initials-based fallback cards,
never fabricated data.

## Training at 32M-row scale — use a Google Colab GPU

`notebooks/00_colab_train_ml32m.ipynb` is the intended way to run
`scripts/train_and_evaluate.py` for real:

1. Open the notebook in Google Colab, set the runtime to a **GPU** (T4 free tier is
   enough), mount Google Drive.
2. Upload/clone this project into the Colab session.
3. The notebook downloads ml-32m (or reuses a cached copy from Drive), runs the full
   training + evaluation pipeline, and copies the trained artifacts back to Drive.
4. Pull `artifacts/models/ml-32m/` and `docs/*.json` down to your local project —
   the Streamlit app loads them instantly instead of retraining.

Only the **neural (NCF)** model actually needs the GPU — popularity, content-based,
and collaborative filtering are all built on `scipy.sparse` matrices and fit in well
under a minute on CPU even at 32M rows. The neural trainer checkpoints its best
weights to `artifacts/models/ml-32m/neural_ncf.pt` every time validation RMSE
improves, so a Colab disconnect mid-training doesn't lose progress.

## Project layout

```
src/
  config.py               single source of truth for paths/hyperparameters (ml-32m only)
  data/
    loader.py              memory-efficient CSV loading (chunked reader for 32M rows)
    validator.py            schema validation + real dataset_summary.json
    cleaner.py              dedupe, drop invalid rows, dense ID mappings
    splitter.py             time-aware, per-user, leakage-free train/val/test
  models/
    popularity.py, content_based.py, collaborative.py, hybrid.py, neural.py
    (popularity/content_based/collaborative all support .save()/.load() via joblib;
     neural via torch.save/load — used by scripts/train_and_evaluate.py and app/state.py
     so the app never has to retrain on 32M rows itself)
  evaluation/
    metrics.py              Precision/Recall/NDCG/HitRate/MAP @K (pure functions)
  coldstart/
    strategies.py            new-user / unknown-user / sparse-user / new-item
  services/
    tmdb_client.py           cached, retrying, gracefully-degrading TMDB client
app/
  streamlit_app.py          Home (page 1 of 10)
  pages/                    Discover, Recommended For You, Movie Explorer,
                             Movie Details, Similar Movies, New User Onboarding,
                             Model Lab, Evaluation Dashboard, About
  state.py                  cached data/model loading; loads pretrained artifacts
                             from artifacts/models/ml-32m/ if present, else trains live
  components/movie_card.py  reusable movie-grid renderer (TMDB-aware)
  styles/custom.css         cinematic dark theme
scripts/
  download_dataset.py       fetches + extracts ml-32m from GroupLens
  train_and_evaluate.py     end-to-end CLI: fit all 5 models + evaluate + save artifacts
notebooks/
  00_colab_train_ml32m.ipynb  <- run this on a Colab GPU for real training
  01_data_exploration.ipynb, 02_model_comparison.ipynb, 03_cold_start_demo.ipynb
tests/                     pytest suite for every pure-function module
docs/                      generated dataset_summary.json + evaluation_results_ml-32m.json
```

## Design decisions worth knowing (for reviewers)

- **Popularity uses a Bayesian weighted rating**, not a raw average — otherwise one
  5-star vote would outrank 10,000 ratings averaging 4.6.
- **Collaborative filtering never builds a dense user-item matrix.** At ml-32m scale
  (200K+ users × 87K items) that would be tens of GB; a `scipy.sparse` matrix is a
  few hundred MB instead.
- **Content similarity uses exact cosine kNN, not a dense N×N matrix** — 87,585²
  floats would be ~30GB.
- **Time-aware, per-user split** — splitting globally by date would starve
  early-joining users of test data; splitting randomly would leak future ratings
  into training.
- **Every recommender exposes the same `get_user_recommendations(user_id, k)`
  signature** so the hybrid layer, evaluator, and Streamlit pages can call any of
  them polymorphically.
- **TMDB is additive, never required.** `TMDBClient.is_available()` gates every
  live call; cache-or-None is always the fallback, and the app is tested (see
  `tests/test_tmdb_client.py`) to keep working with zero API key.
- **No claim of "hybrid/neural is best" is hard-coded anywhere** — the Evaluation
  Dashboard reads `docs/evaluation_results_ml-32m.json`, produced by actually running
  `evaluate_model()`, and reports whichever model measures highest.
- **Trained artifacts are persisted, not just the neural checkpoint** — popularity,
  content-based, and collaborative filtering all save/load via `joblib` so the
  Streamlit app doesn't refit a model on 32M ratings on every cold start.

## Honest limitations

- The neural (NCF) model is trained as **rating regression (MSE)**, which is not
  the same objective as ranking quality — check the Evaluation Dashboard rather
  than assuming it wins on NDCG/Precision@K.
- Content-based similarity uses only genres + user tags (the only text-like fields
  in raw MovieLens) unless TMDB `overview` enrichment is wired into the corpus.
- **Hardware for a full ml-32m run**: schema validation and the sparse/joblib-backed
  popularity, content-based, and collaborative models were verified end-to-end
  against the real 32,000,204-row file. The neural model is the one step that
  genuinely wants a GPU at this row count — see "Training at 32M-row scale" above.
  A machine with **16GB+ RAM** is recommended for the full in-memory pipeline
  (`clean_pipeline`'s sort+dedupe, sparse SVD, TF-IDF, and NCF training all holding
  the frame in memory at once); a modest cloud VM or Colab's standard runtime is
  enough for everything except the neural GPU step.

## Testing

```bash
pytest tests/ -v
```

15 tests covering evaluation metrics, data cleaning/splitting invariants (no
time-leakage, dense ID bijectivity), and TMDB offline-degradation behavior.

## Data attribution

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4,
Article 19. MovieLens data used under GroupLens' non-commercial research terms;
this project is a portfolio/educational demonstration.

---
Built by Rahat — [github.com/Rm3997](https://github.com/Rm3997)
