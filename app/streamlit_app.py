"""
CineSignal -- Home page.
Run:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.state import load_css, load_raw_data, get_trained_models, load_dataset_summary, get_tmdb_client
from app.components.movie_card import render_movie_grid
from src import config

st.set_page_config(page_title="CineSignal", page_icon="\U0001F3AC", layout="wide", initial_sidebar_state="expanded")
load_css()

with st.sidebar:
    st.markdown("### \U0001F3AC CineSignal")
    st.caption(f"Dataset: **{config.DATASET}** (32M real MovieLens ratings)")
    st.markdown("---")
    st.page_link("streamlit_app.py", label="Home", icon="\U0001F3E0")
    st.page_link("pages/2_Discover.py", label="Discover", icon="\U0001F9ED")
    st.page_link("pages/3_Recommended_For_You.py", label="Recommended For You", icon="\u2728")
    st.page_link("pages/4_Movie_Explorer.py", label="Movie Explorer", icon="\U0001F50D")
    st.page_link("pages/5_Movie_Details.py", label="Movie Details", icon="\U0001F3A5")
    st.page_link("pages/6_Similar_Movies.py", label="Similar Movies", icon="\U0001F517")
    st.page_link("pages/7_Cold_Start.py", label="New User Onboarding", icon="\U0001F195")
    st.page_link("pages/8_Model_Lab.py", label="Model Lab", icon="\U0001F9EA")
    st.page_link("pages/9_Evaluation_Dashboard.py", label="Evaluation Dashboard", icon="\U0001F4CA")
    st.page_link("pages/10_About.py", label="About", icon="\u2139\uFE0F")

if not config.RATINGS_CSV.exists():
    st.error(
        f"Dataset folder data/raw/{config.DATASET}/ not found. Run "
        f"`python scripts/download_dataset.py` first, or extract the ml-32m "
        f"archive there yourself."
    )
    st.stop()

with st.spinner("Loading..."):
    ratings, movies, tags, links = load_raw_data()
    models = get_trained_models()
    tmdb = get_tmdb_client()

pop = models["popularity"]
featured = pop.recommend(k=5)
featured = featured.merge(movies, on="movieId", how="left")
hero_title = featured.iloc[0]["clean_title"] if len(featured) else "CineSignal"

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-badge">REAL MOVIELENS DATA \u00b7 {config.DATASET}</div>
        <div class="hero-title">Movies you'll actually<br/>want to watch.</div>
        <div class="hero-sub">
            Popularity, collaborative filtering, content-based, hybrid and neural
            recommenders -- all trained on real user ratings, evaluated on the
            same held-out test set, with every score explained.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
for col, (label, value) in zip(
    [c1, c2, c3, c4],
    [
        ("Ratings", f"{len(ratings):,}"),
        ("Movies", f"{movies['movieId'].nunique():,}"),
        ("Users", f"{ratings['userId'].nunique():,}"),
        ("Timespan", f"{ratings['ts'].dt.year.min()}\u2013{ratings['ts'].dt.year.max()}"),
    ],
):
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
st.markdown("### Featured this week")
st.caption("Ranked by a confidence-weighted popularity score (not a raw average) \u2014 see Model Lab for the formula.")
render_movie_grid(featured, tmdb_client=tmdb, links_df=links, columns=5,
                   score_col="weighted_rating", score_label="Score")

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
st.markdown("### How the recommendation engine works")
o1, o2, o3, o4, o5 = st.columns(5)
for col, (name, desc) in zip(
    [o1, o2, o3, o4, o5],
    [
        ("Popularity", "Confidence-weighted rating, not raw average."),
        ("Collaborative", "Matrix factorization on real rating patterns."),
        ("Content-Based", "TF-IDF over genres & tags, cosine similarity."),
        ("Hybrid", "Configurable blend of collaborative + content."),
        ("Neural", "PyTorch embedding-MLP (NCF) trained on ratings."),
    ],
):
    with col:
        st.markdown(f"**{name}**")
        st.caption(desc)

st.info("Head to **Recommended For You** to try personalized recommendations, or **New User Onboarding** "
        "if you don't have a user ID yet.")
