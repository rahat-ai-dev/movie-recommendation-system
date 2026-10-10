import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.state import load_css, load_raw_data, get_trained_models
from src import config

st.set_page_config(page_title="Model Lab \u2014 CineSignal", page_icon="\U0001F9EA", layout="wide")
load_css()
st.markdown("## \U0001F9EA Model Lab")
st.caption("Technical detail for engineers / recruiters. Every number below is measured, not asserted.")

ratings, movies, tags, links = load_raw_data()
models = get_trained_models()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Popularity", "Collaborative", "Content-Based", "Hybrid", "Neural (NCF)"])

with tab1:
    pop = models["popularity"]
    st.markdown(f"""
**Formula:** `WR = (v/(v+m))*R + (m/(v+m))*C`
- `v` = number of ratings for the movie, `m` = {pop.m} (min-votes threshold)
- `R` = movie's own mean rating, `C` = catalog mean = **{pop.global_mean_:.3f}**
- Movies with fewer than `min_ratings={pop.min_ratings}` votes are excluded from ranking entirely.

**Why not raw average?** A movie with one 5-star rating would otherwise outrank a
movie with 10,000 ratings averaging 4.6 \u2014 this Bayesian shrinkage pulls low-vote
items toward the catalog mean.

**Eligible catalog:** {len(pop.ranking_):,} movies out of {movies['movieId'].nunique():,} total.
    """)

with tab2:
    cf = models["collaborative"]
    st.markdown(f"""
**Method:** truncated SVD (`scipy.sparse.linalg.svds`) on a **sparse**, per-user
mean-centered user-item rating matrix. No dense matrix is ever built.

- Users: {cf.n_users_:,} \u00b7 Items: {cf.n_items_:,} \u00b7 Latent factors: {cf.user_factors_.shape[1]}
- Sparsity: {100*(1 - len(ratings)/(cf.n_users_*cf.n_items_)):.3f}% of the user-item matrix is empty
- Scoring: `user_mean + user_factors \u00b7 item_factors`, ranked, seen items excluded

**Limitations:** no cold-start support for brand-new users/items (handled separately
by the Cold-Start module); treats unobserved entries as centered-zero, a standard
approximation for implicit matrix factorization on explicit ratings.
    """)

with tab3:
    cb = models["content_based"]
    st.markdown(f"""
**Method:** TF-IDF over real genres + aggregated user tags \u2192 exact cosine
nearest-neighbors (`sklearn.NearestNeighbors`, no dense N\u00d7N similarity matrix,
which would need ~{movies['movieId'].nunique()**2 / 1e9:.2f}B floats at this catalog size).

- Vocabulary size: {cb.matrix_.shape[1]:,} terms
- TF-IDF matrix shape: {cb.matrix_.shape[0]:,} movies \u00d7 {cb.matrix_.shape[1]:,} features (sparse)

**Limitation:** no plot/overview text in raw MovieLens \u2014 similarity is genre+tag
based only, until TMDB `overview` enrichment is wired into the corpus.
    """)

with tab4:
    st.markdown(f"""
**Formula:** `hybrid_score = alpha \u00d7 collaborative_norm + (1-alpha) \u00d7 content_norm`

Both components are min-max normalized per-request before blending (CF scores and
cosine similarities live on different scales). Default alpha = **{config.HYBRID_DEFAULT_ALPHA}**,
configurable per-request on the Recommended For You page.

Falls back to the Popularity baseline automatically for any user unseen during
training.
    """)

with tab5:
    neural = models["neural"]
    st.markdown(f"""
**Architecture:** Embedding-MLP Neural Collaborative Filtering (PyTorch)

```
UserID -> Embedding({neural.embed_dim}) -\\
                                          concat -> Dense{config.NEURAL_HIDDEN_LAYERS} (ReLU+Dropout {config.NEURAL_DROPOUT}) -> Linear(1)
MovieID -> Embedding({neural.embed_dim}) -/
```

- Optimizer: Adam, lr={neural.lr}, batch_size={neural.batch_size}
- Early stopping patience: {neural.patience} epochs on validation RMSE
- Device: **{neural.device}**
    """)
    if hasattr(neural, "history_"):
        import pandas as pd
        hist = pd.DataFrame(neural.history_)
        st.line_chart(hist.set_index("epoch")[["train_rmse", "val_rmse"]])
    st.caption("Honest note: the neural model is trained as rating regression (MSE), which is not the same "
               "objective as ranking quality \u2014 check the Evaluation Dashboard to see how it actually "
               "compares to collaborative filtering on NDCG/Precision@K rather than assuming it wins.")
