"""
Shared, cached application state. Streamlit re-runs the whole script on
every interaction, so every expensive object (dataframes, fitted models)
is wrapped in st.cache_data / st.cache_resource -- computed once, then
reused instantly.

Single dataset: ml-32m. Model artifacts are loaded from
artifacts/models/ml-32m/ if present (produced by
`python scripts/train_and_evaluate.py`, ideally run once on a GPU via
notebooks/00_colab_train_ml32m.ipynb) -- this avoids re-fitting a
collaborative/content/neural model on 32M ratings inside the Streamlit
process itself. If no saved artifacts exist yet, it falls back to training
live (slow at full 32M scale without a GPU, but keeps the app usable).
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src import config


@st.cache_data(show_spinner=False)
def load_dataset_summary() -> dict:
    path = config.PROJECT_ROOT / "docs" / "dataset_summary.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


@st.cache_resource(show_spinner="Loading MovieLens ml-32m data...")
def load_raw_data():
    from src.data import loader
    from src.data.cleaner import clean_pipeline

    cache_path = config.PROCESSED_DATASET_DIR / "ratings_clean.parquet"
    if cache_path.exists():
        ratings = pd.read_parquet(cache_path)
    else:
        ratings = clean_pipeline(loader.load_ratings())
        ratings.to_parquet(cache_path, index=False)  # cache so future restarts skip dedupe+sort

    movies = loader.load_movies()
    tags = loader.load_tags()
    links = loader.load_links()
    return ratings, movies, tags, links


@st.cache_resource(show_spinner="Loading trained recommendation models...")
def get_trained_models():
    """Load pretrained artifacts if available (recommended path for
    ml-32m); otherwise fit live as a fallback so the app still works
    without a prior training run."""
    from src.data.splitter import train_val_test_split
    from src.models.popularity import PopularityRecommender
    from src.models.content_based import ContentBasedRecommender
    from src.models.collaborative import CollaborativeRecommender
    from src.models.hybrid import HybridRecommender
    from src.models.neural import NeuralRecommender
    from src.coldstart.strategies import ColdStartManager

    ratings, movies, tags, links = load_raw_data()
    train, val, test = train_val_test_split(ratings)

    d = config.MODEL_DATASET_DIR
    pop_path, cb_path, cf_path, nn_path = (
        d / "popularity.joblib", d / "content_based.joblib",
        d / "collaborative.joblib", d / "neural_ncf.pt",
    )

    used_pretrained = pop_path.exists() and cb_path.exists() and cf_path.exists()
    if used_pretrained:
        pop = PopularityRecommender.load(pop_path)
        cb = ContentBasedRecommender.load(cb_path)
        cf = CollaborativeRecommender.load(cf_path)
    else:
        pop = PopularityRecommender().fit(train)
        cb = ContentBasedRecommender().fit(movies, tags)
        cf = CollaborativeRecommender().fit(train)

    hybrid = HybridRecommender(cf, cb, pop)
    coldstart = ColdStartManager(cb, pop)

    if nn_path.exists():
        neural = NeuralRecommender.load(nn_path)
        neural._seen = train.groupby("userId")["movieId"].apply(set).to_dict()
    else:
        # Live fallback: a light-touch model so the page still renders;
        # for a real ml-32m neural model, train via the Colab notebook.
        neural = NeuralRecommender(epochs=3)
        neural.fit(train, val, verbose=False)

    return {
        "ratings": ratings, "movies": movies, "tags": tags, "links": links,
        "train": train, "val": val, "test": test,
        "popularity": pop, "content_based": cb, "collaborative": cf,
        "hybrid": hybrid, "neural": neural, "coldstart": coldstart,
        "used_pretrained_artifacts": used_pretrained,
    }


@st.cache_resource(show_spinner=False)
def get_tmdb_client():
    from src.services.tmdb_client import TMDBClient
    api_key = ""
    try:
        api_key = st.secrets.get("TMDB_API_KEY", "") if hasattr(st, "secrets") else ""
    except Exception:
        api_key = ""  # no secrets.toml configured -- fall back to env var only
    api_key = api_key or config.TMDB_API_KEY
    return TMDBClient(api_key=api_key)


def load_css():
    css_path = config.PROJECT_ROOT / "app" / "styles" / "custom.css"
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)