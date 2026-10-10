import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.state import load_css, load_raw_data, get_trained_models, get_tmdb_client
from app.components.movie_card import render_movie_grid

st.set_page_config(page_title="Similar Movies \u2014 CineSignal", page_icon="\U0001F517", layout="wide")
load_css()
st.markdown("## \U0001F517 Similar Movies")
st.caption("Movie \u2192 Content Representation (TF-IDF genres/tags) \u2192 Cosine Similarity \u2192 Top-K")

ratings, movies, tags, links = load_raw_data()
models = get_trained_models()
tmdb = get_tmdb_client()

title_choice = st.selectbox(
    "Anchor movie",
    movies.sort_values("clean_title")["clean_title"] + " (" + movies["movieId"].astype(str) + ")",
)
movie_id = int(title_choice.rsplit("(", 1)[1].rstrip(")"))
k = st.slider("How many similar movies", 5, 25, 12)

similar = models["content_based"].get_similar_movies(movie_id, k=k).merge(movies, on="movieId", how="left")
render_movie_grid(similar, tmdb_client=tmdb, links_df=links, columns=5, score_col="similarity", score_label="Similarity")
