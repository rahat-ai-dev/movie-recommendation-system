"""
Central configuration for the recommendation system.

Single-dataset build: ml-32m (32M MovieLens ratings) is the only supported
dataset. No other file should hard-code a dataset name or path -- they all
read from this module.

Loads a project-root `.env` file (if present) BEFORE reading any os.environ
value below, so `TMDB_API_KEY=...` in `.env` is actually picked up by
`streamlit run` / plain `python` invocations, not just shells that already
exported it.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------- paths ---
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
TMDB_CACHE_DIR = ARTIFACTS_DIR / "tmdb_cache"

for _d in (DATA_PROCESSED_DIR, MODELS_DIR, TMDB_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------- dataset ---
DATASET = "ml-32m"

RAW_DATASET_DIR = DATA_RAW_DIR / DATASET
PROCESSED_DATASET_DIR = DATA_PROCESSED_DIR / DATASET
PROCESSED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DATASET_DIR = MODELS_DIR / DATASET
MODEL_DATASET_DIR.mkdir(parents=True, exist_ok=True)

RATINGS_CSV = RAW_DATASET_DIR / "ratings.csv"
MOVIES_CSV = RAW_DATASET_DIR / "movies.csv"
TAGS_CSV = RAW_DATASET_DIR / "tags.csv"
LINKS_CSV = RAW_DATASET_DIR / "links.csv"

# Official GroupLens download -- used by scripts/download_dataset.py and the
# Colab notebook when data/raw/ml-32m/ is empty.
ML32M_DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"

# Dataset facts (filled in by data/validator.py at inspection time, used by
# app + docs so numbers are never hard-coded/guessed).
DATASET_RELEASE_DATES = {
    "ml-32m": "2023-10-13 (last updated)",
}

# ---------------------------------------------------- split / evaluation ---
TIME_SPLIT_TEST_FRACTION = 0.2   # last 20% of EACH user's timeline -> test
MIN_INTERACTIONS_FOR_TEST_USER = 5  # users with fewer ratings skip eval split
TOP_K_DEFAULT = 10
EVAL_K_VALUES = (5, 10, 20)

# --------------------------------------------------------- popularity ---
POPULARITY_MIN_RATINGS = 20        # Bayesian-average style minimum votes
POPULARITY_CONFIDENCE_M = 20       # "m" in the weighted-rating formula

# ---------------------------------------------------------- content-based ---
TFIDF_MAX_FEATURES = 20000
CONTENT_NEIGHBORS = 50             # ANN neighbor pool size

# ------------------------------------------------------ collaborative (MF) ---
CF_LATENT_FACTORS = 50
CF_RANDOM_STATE = 42

# --------------------------------------------------------------- hybrid ---
HYBRID_DEFAULT_ALPHA = 0.5   # weight on collaborative score; (1-alpha) content

# --------------------------------------------------------------- neural ---
NEURAL_EMBED_DIM = 32
NEURAL_HIDDEN_LAYERS = (128, 64, 32)
NEURAL_DROPOUT = 0.2
NEURAL_LR = 1e-3
NEURAL_BATCH_SIZE = 16384         # larger default -- fewer, bigger batches means
                                   # less fixed per-step Python overhead relative
                                   # to actual compute, which matters a lot on CPU
                                   # and costs nothing on GPU at this model's size
NEURAL_EPOCHS = 15
NEURAL_EARLY_STOP_PATIENCE = 3
NEURAL_SEED = 42

# ------------------------------------------------------------------ tmdb ---
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"
TMDB_POSTER_SIZE = "w342"
TMDB_BACKDROP_SIZE = "w780"
TMDB_TIMEOUT_SECONDS = 5
TMDB_MAX_RETRIES = 2
TMDB_CACHE_TTL_DAYS = 30
