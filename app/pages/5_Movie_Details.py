import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.state import load_css, load_raw_data, get_trained_models, get_tmdb_client
from app.components.movie_card import render_movie_grid

st.set_page_config(page_title="Movie Details \u2014 CineSignal", page_icon="\U0001F3A5", layout="wide")
load_css()
st.markdown("## \U0001F3A5 Movie Details")

ratings, movies, tags, links = load_raw_data()
models = get_trained_models()
tmdb = get_tmdb_client()

title_choice = st.selectbox(
    "Pick a movie",
    movies.sort_values("clean_title")["clean_title"] + " (" + movies["movieId"].astype(str) + ")",
)
movie_id = int(title_choice.rsplit("(", 1)[1].rstrip(")"))
row = movies[movies.movieId == movie_id].iloc[0]

link_row = links[links.movieId == movie_id]
tmdb_meta = None
if tmdb.is_available() and len(link_row) and not link_row.iloc[0].isna()["tmdbId"]:
    tmdb_meta = tmdb.get_movie_metadata(int(link_row.iloc[0]["tmdbId"]))

backdrop = tmdb_meta.get("backdrop_url") if tmdb_meta else None
bg_style = f'background-image: url({backdrop});' if backdrop else ""
year_str = f" ({int(row['year'])})" if row["year"] == row["year"] else ""

st.markdown(
    f"""<div class="hero" style="{bg_style}">
        <div class="hero-title" style="font-size:2.4rem;">{row['clean_title']}{year_str}</div>
        <div class="hero-sub">{row['genres'].replace('|', ' \u00b7 ')}</div>
    </div>""",
    unsafe_allow_html=True,
)

pop_row = models["popularity"].ranking_[models["popularity"].ranking_.movieId == movie_id]
c1, c2, c3, c4 = st.columns(4)
with c1:
    n = int(pop_row["num_ratings"].iloc[0]) if len(pop_row) else int((ratings.movieId == movie_id).sum())
    st.markdown(f'<div class="metric-card"><div class="metric-value">{n}</div><div class="metric-label">MovieLens ratings</div></div>', unsafe_allow_html=True)
with c2:
    avg = ratings[ratings.movieId == movie_id]["rating"].mean()
    avg_display = f"{avg:.2f}" if avg == avg else "\u2014"
    st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_display}</div><div class="metric-label">Avg rating</div></div>', unsafe_allow_html=True)
with c3:
    if tmdb_meta and tmdb_meta.get("vote_average") is not None:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{tmdb_meta["vote_average"]:.1f}</div><div class="metric-label">TMDB score</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="metric-card"><div class="metric-value">\u2014</div><div class="metric-label">TMDB score</div></div>', unsafe_allow_html=True)
with c4:
    runtime = tmdb_meta.get("runtime") if tmdb_meta else None
    st.markdown(f'<div class="metric-card"><div class="metric-value">{runtime or "\u2014"}</div><div class="metric-label">Runtime (min)</div></div>', unsafe_allow_html=True)

if tmdb_meta and tmdb_meta.get("overview"):
    st.markdown("#### Overview")
    st.write(tmdb_meta["overview"])
elif not tmdb.is_available():
    st.caption("Connect TMDB_API_KEY to show a plot overview, runtime, and poster/backdrop art here.")

movie_tags = tags[tags.movieId == movie_id]["tag"].value_counts()
if len(movie_tags):
    st.markdown("#### Real user tags")
    st.write(", ".join(movie_tags.index[:15]))

st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
st.markdown("### You may also like")
similar = models["content_based"].get_similar_movies(movie_id, k=10).merge(movies, on="movieId", how="left")
render_movie_grid(similar, tmdb_client=tmdb, links_df=links, columns=5, score_col="similarity", score_label="Similarity")
