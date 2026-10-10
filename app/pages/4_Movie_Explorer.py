import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.state import load_css, load_raw_data, get_trained_models, get_tmdb_client
from app.components.movie_card import render_movie_grid

st.set_page_config(page_title="Movie Explorer \u2014 CineSignal", page_icon="\U0001F50D", layout="wide")
load_css()
st.markdown("## \U0001F50D Movie Explorer")

ratings, movies, tags, links = load_raw_data()
models = get_trained_models()
tmdb = get_tmdb_client()
pop_stats = models["popularity"].ranking_[["movieId", "num_ratings", "avg_rating"]]
catalog = movies.merge(pop_stats, on="movieId", how="left")

c1, c2, c3, c4, c5 = st.columns([2, 1.2, 1, 1, 1.2])
with c1:
    query = st.text_input("Search title", "")
with c2:
    all_genres = sorted({g for gs in movies["genres_list"] for g in gs})
    genre_filter = st.multiselect("Genre", all_genres)
with c3:
    min_year, max_year = int(movies["year"].min(skipna=True) or 1900), int(movies["year"].max(skipna=True) or 2020)
    year_range = st.slider("Year", min_year, max_year, (min_year, max_year))
with c4:
    min_rating = st.slider("Min avg rating", 0.0, 5.0, 0.0, 0.5)
with c5:
    sort_by = st.selectbox("Sort by", ["Popularity", "Avg rating", "Title A-Z", "Newest"])

filtered = catalog.copy()
if query:
    filtered = filtered[filtered["clean_title"].str.contains(query, case=False, na=False)]
if genre_filter:
    filtered = filtered[filtered["genres_list"].apply(lambda gs: any(g in gs for g in genre_filter))]
filtered = filtered[filtered["year"].between(year_range[0], year_range[1]) | filtered["year"].isna()]
filtered = filtered[filtered["avg_rating"].fillna(0) >= min_rating]

if sort_by == "Popularity":
    filtered = filtered.sort_values("num_ratings", ascending=False, na_position="last")
elif sort_by == "Avg rating":
    filtered = filtered.sort_values("avg_rating", ascending=False, na_position="last")
elif sort_by == "Title A-Z":
    filtered = filtered.sort_values("clean_title")
else:
    filtered = filtered.sort_values("year", ascending=False, na_position="last")

st.caption(f"{len(filtered):,} movies match your filters.")
page_size = 20
page = st.number_input("Page", min_value=1, max_value=max(1, (len(filtered) - 1) // page_size + 1), value=1)
page_df = filtered.iloc[(page - 1) * page_size: page * page_size]

render_movie_grid(page_df, tmdb_client=tmdb, links_df=links, columns=5,
                   score_col="avg_rating", score_label="Avg \u2605", explanation_col=None)

st.info("Click through to **Movie Details** (use the movie's ID) for the full cinematic view and recommendations.")
