import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.state import load_css, load_raw_data, get_trained_models, get_tmdb_client
from app.components.movie_card import render_movie_grid

st.set_page_config(page_title="New User Onboarding \u2014 CineSignal", page_icon="\U0001F195", layout="wide")
load_css()
st.markdown("## \U0001F195 Tell us what you like")
st.markdown('<span class="badge badge-coldstart">Cold-start \u00b7 no fake ratings, real catalog only</span>',
            unsafe_allow_html=True)

ratings, movies, tags, links = load_raw_data()
models = get_trained_models()
tmdb = get_tmdb_client()
coldstart = models["coldstart"]

mode = st.radio("Onboard by", ["Pick movies I like", "Pick genres I like"], horizontal=True)

if mode == "Pick movies I like":
    popular_pool = models["popularity"].recommend(k=60).merge(movies, on="movieId", how="left")
    options = (popular_pool["clean_title"] + " (" + popular_pool["movieId"].astype(str) + ")").tolist()
    picks = st.multiselect("Select a few movies you enjoy (chosen from real, popular catalog titles):", options)
    liked_ids = [int(p.rsplit("(", 1)[1].rstrip(")")) for p in picks]

    if st.button("Build My Recommendations", type="primary", disabled=not liked_ids):
        recs = coldstart.new_user_from_selections(liked_ids, k=15).merge(movies, on="movieId", how="left")
        st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
        render_movie_grid(recs, tmdb_client=tmdb, links_df=links, columns=5,
                           score_col="final_score", score_label="Match")
else:
    all_genres = sorted({g for gs in movies["genres_list"] for g in gs})
    liked_genres = st.multiselect("Pick genres you enjoy:", all_genres)
    if st.button("Build My Recommendations", type="primary", disabled=not liked_genres):
        recs = coldstart.new_user_from_genres(liked_genres, movies, k=15).merge(
            movies, on="movieId", how="left", suffixes=("", "_dup")
        )
        st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
        render_movie_grid(recs, tmdb_client=tmdb, links_df=links, columns=5,
                           score_col="weighted_rating", score_label="Score")

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
with st.expander("How cold-start works here"):
    st.markdown(
        """
- **New user (this page):** your real selections build a content preference profile
  (TF-IDF over genres/tags) \u2192 nearest-neighbor retrieval \u2192 re-ranked by popularity.
- **Unknown user (no ID at all):** falls back straight to the popularity baseline.
- **Sparse user (very few ratings):** the Hybrid recommender lowers its collaborative
  weight (alpha) so noisy CF signal doesn't dominate.
- **New item (no interactions yet):** served purely through content-based similarity,
  since collaborative filtering has nothing to learn from yet.

No synthetic ratings or fake users are ever created \u2014 only real catalog items and
your real selections feed these recommendations.
        """
    )
