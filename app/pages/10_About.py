import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.state import load_css

st.set_page_config(page_title="About \u2014 CineSignal", page_icon="\u2139\uFE0F", layout="wide")
load_css()
st.markdown("## \u2139\uFE0F About CineSignal")

st.markdown(
    """
**CineSignal** is a portfolio-grade movie recommendation platform built entirely on
real MovieLens data (GroupLens Research) -- no synthetic users, ratings, or metadata.

### Data sources & honesty
- **ml-32m** (GroupLens/MovieLens): 32,000,204 ratings, 87,585 movies, 200,948 users
  \u2014 last updated 2023-10-13. The one and only dataset this build trains and
  evaluates on \u2014 no small dev dataset, no shortcuts.
- MovieLens interaction data does **not** extend into 2024\u20132026. Where the app shows
  "Current Movies", that content comes from **live TMDB metadata**, explicitly
  labeled and never blended with historical personalized recommendations.

### Architecture
`data/` (raw MovieLens ml-32m) \u2192 `src/data/` (validate \u2192 clean \u2192 split) \u2192
`src/models/` (5 recommenders) \u2192 `src/evaluation/` (ranking metrics) \u2192
`app/` (this Streamlit UI). `src/config.py` is the single source of truth for
every path and hyperparameter \u2014 no model code duplicates a path or a magic number.

### Training at this scale
32M ratings is GPU territory for the neural model. Train once with
`notebooks/00_colab_train_ml32m.ipynb` on a free Google Colab GPU runtime, then
drop the saved artifacts into `artifacts/models/ml-32m/` \u2014 the app loads them
instantly instead of retraining on every run.

### Recommenders implemented
Popularity (Bayesian-weighted) \u00b7 Collaborative Filtering (sparse SVD matrix
factorization) \u00b7 Content-Based (TF-IDF + cosine kNN) \u00b7 Hybrid (configurable blend)
\u00b7 Neural (PyTorch embedding-MLP NCF).

### Built by
Rahat \u2014 [github.com/Rm3997](https://github.com/Rm3997)
    """
)
