import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.state import load_css, load_raw_data, get_trained_models, get_tmdb_client
from app.components.movie_card import render_movie_grid

st.set_page_config(page_title="Discover \u2014 CineSignal", page_icon="\U0001F9ED", layout="wide")
load_css()
st.markdown("## \U0001F9ED Discover")

ratings, movies, tags, links = load_raw_data()
models = get_trained_models()
tmdb = get_tmdb_client()
pop, cb = models["popularity"], models["content_based"]

tab_pop, tab_top, tab_genre, tab_current = st.tabs(
    ["\U0001F525 Popular", "\u2b50 Top Rated", "\U0001F3AD By Genre", "\U0001F195 Current Movies (TMDB)"]
)

with tab_pop:
    st.caption("Data source: MovieLens historical ratings \u2014 confidence-weighted popularity.")
    st.markdown('<span class="badge badge-historical">Historical MovieLens data</span>', unsafe_allow_html=True)
    top = pop.recommend(k=15).merge(movies, on="movieId", how="left")
    render_movie_grid(top, tmdb_client=tmdb, links_df=links, columns=5,
                       score_col="weighted_rating", score_label="Score")

with tab_top:
    st.caption("Highest raw average rating among movies with enough votes to be meaningful.")
    st.markdown('<span class="badge badge-historical">Historical MovieLens data</span>', unsafe_allow_html=True)
    top_rated = pop.ranking_.sort_values("avg_rating", ascending=False).head(15).merge(movies, on="movieId", how="left")
    render_movie_grid(top_rated, tmdb_client=tmdb, links_df=links, columns=5,
                       score_col="avg_rating", score_label="Avg \u2605")

with tab_genre:
    all_genres = sorted({g for gs in movies["genres_list"] for g in gs})
    genre = st.selectbox("Genre", all_genres)
    mask = movies["genres_list"].apply(lambda gs: genre in gs)
    genre_ids = movies.loc[mask, "movieId"]
    ranked = pop.ranking_[pop.ranking_["movieId"].isin(genre_ids)].head(15).merge(movies, on="movieId", how="left")
    st.markdown('<span class="badge badge-historical">Historical MovieLens data</span>', unsafe_allow_html=True)
    render_movie_grid(ranked, tmdb_client=tmdb, links_df=links, columns=5,
                       score_col="weighted_rating", score_label="Score")

with tab_current:
    st.markdown('<span class="badge badge-current">Current TMDB metadata (not historical interactions)</span>',
                unsafe_allow_html=True)
    st.caption("MovieLens does not contain 2024\u20132026 interaction data. This section shows CURRENT releases "
               "from TMDB's live catalog, kept explicitly separate from personalized/historical recommendations.")
    if not tmdb.is_available():
        st.markdown(
            '<div class="empty-state">TMDB_API_KEY not configured \u2014 current-movie discovery needs a live '
            'TMDB connection. Set TMDB_API_KEY in .env or Streamlit secrets to enable this section.</div>',
            unsafe_allow_html=True,
        )
    else:
        now_playing = tmdb.get_now_playing()
        if not now_playing:
            st.markdown('<div class="empty-state">TMDB unreachable and no cached results yet.</div>', unsafe_allow_html=True)
        else:
            cols = st.columns(5)
            for i, m in enumerate(now_playing[:15]):
                from app.components.movie_card import render_movie_card
                with cols[i % 5]:
                    render_movie_card(
                        title=m.get("title", "Unknown"),
                        rating=m.get("vote_average"),
                        poster_url=tmdb.poster_url(m.get("poster_path")),
                    )
