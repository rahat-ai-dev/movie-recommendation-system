import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.state import load_css, load_raw_data, get_trained_models, get_tmdb_client
from app.components.movie_card import render_movie_grid
from src import config

st.set_page_config(page_title="Recommended For You \u2014 CineSignal", page_icon="\u2728", layout="wide")
load_css()
st.markdown("## \u2728 Recommended For You")
st.markdown('<span class="badge badge-historical">Personalized \u00b7 based on real historical ratings</span>',
            unsafe_allow_html=True)

ratings, movies, tags, links = load_raw_data()
models = get_trained_models()
tmdb = get_tmdb_client()

valid_users = sorted(ratings["userId"].unique().tolist())

c1, c2, c3, c4 = st.columns([1.2, 1.4, 0.8, 1])
with c1:
    user_id = st.selectbox("User", valid_users, index=0)
with c2:
    method = st.selectbox("Method", ["Popularity", "Collaborative Filtering", "Content-Based", "Hybrid", "Neural"])
with c3:
    k = st.slider("Top-K", 5, 30, 10)
with c4:
    alpha = st.slider("Hybrid alpha (CF weight)", 0.0, 1.0, config.HYBRID_DEFAULT_ALPHA, 0.05,
                       disabled=(method != "Hybrid"))

score_col, score_label = None, "Score"

if method == "Popularity":
    recs = models["popularity"].get_user_recommendations(user_id, k=k)
    score_col, score_label = "weighted_rating", "Pop. score"
elif method == "Collaborative Filtering":
    recs = models["collaborative"].get_user_recommendations(user_id, k=k)
    score_col, score_label = "cf_score", "CF score"
elif method == "Content-Based":
    liked = list(models["collaborative"]._seen.get(user_id, set()))[:20]
    recs = models["content_based"].get_recommendations_for_profile(liked, k=k) if liked else models["popularity"].get_user_recommendations(user_id, k=k)
    score_col, score_label = "similarity", "Similarity"
elif method == "Hybrid":
    recs = models["hybrid"].get_user_recommendations(user_id, k=k, alpha=alpha)
    score_col, score_label = "hybrid_score", "Hybrid score"
else:  # Neural
    recs = models["neural"].get_user_recommendations(user_id, k=k)
    score_col, score_label = "neural_score", "Neural score"

recs = recs.merge(movies, on="movieId", how="left")
st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
st.caption(f"Showing {method} recommendations for user {user_id} \u2014 already-rated movies excluded.")
render_movie_grid(recs, tmdb_client=tmdb, links_df=links, columns=5, score_col=score_col, score_label=score_label)

with st.expander("This user's rating history (real data)"):
    hist = ratings[ratings.userId == user_id].merge(movies, on="movieId").sort_values("rating", ascending=False)
    st.dataframe(hist[["clean_title", "genres", "rating", "ts"]].head(30), width='stretch', hide_index=True)
