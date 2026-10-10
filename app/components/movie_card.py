from __future__ import annotations

import html

import pandas as pd
import streamlit as st


def _poster_block(poster_url: str | None, title: str) -> str:
    if poster_url:
        return f'<img src="{poster_url}" style="width:100%;aspect-ratio:2/3;object-fit:cover;display:block;" />'
    initials = "".join(w[0] for w in title.split()[:2]).upper() if isinstance(title, str) and title else "?"
    return f'<div class="movie-poster-fallback">{html.escape(initials)}</div>'


def render_movie_card(title: str, year=None, genres: str = "", rating: float | None = None,
                       score: float | None = None, score_label: str = "Score",
                       explanation: str | None = None, poster_url: str | None = None):
    title = title if isinstance(title, str) else "Unknown"
    year_str = f" ({int(year)})" if pd.notna(year) else ""
    genres = genres if isinstance(genres, str) and genres else ""
    meta_bits = [b for b in [genres, f"\u2605 {rating:.2f}" if pd.notna(rating) else None] if b]
    meta = " &middot; ".join(meta_bits)
    score_html = f'<span class="movie-score">{score_label}: {score:.2f}</span>' if pd.notna(score) else ""
    expl_html = f'<div class="movie-explanation">{html.escape(explanation)}</div>' if isinstance(explanation, str) and explanation else ""

    st.markdown(
        f"""
        <div class="movie-card">
            {_poster_block(poster_url, title)}
            <div class="movie-card-body">
                <div class="movie-title">{html.escape(title)}{year_str}</div>
                <div class="movie-meta">{meta}</div>
                {score_html}
                {expl_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_movie_grid(df, tmdb_client=None, links_df=None, columns: int = 5,
                       score_col: str | None = None, score_label: str = "Score",
                       explanation_col: str | None = "explanation"):
    """df must contain movieId, clean_title/title, year, genres (or genres_list)."""
    if df is None or len(df) == 0:
        st.markdown('<div class="empty-state">No results to show yet.</div>', unsafe_allow_html=True)
        return

    cols = st.columns(columns)
    for i, (_, row) in enumerate(df.iterrows()):
        poster_url = None
        if tmdb_client is not None and links_df is not None and tmdb_client.is_available():
            link_row = links_df[links_df["movieId"] == row["movieId"]]
            if len(link_row) and not link_row.iloc[0].isna()["tmdbId"]:
                meta = tmdb_client.get_movie_metadata(int(link_row.iloc[0]["tmdbId"]))
                if meta:
                    poster_url = meta.get("poster_url")

        title = row.get("clean_title", row.get("title", "Unknown"))
        genres = row.get("genres", "")
        with cols[i % columns]:
            render_movie_card(
                title=title,
                year=row.get("year"),
                genres=genres if isinstance(genres, str) else "",
                score=float(row[score_col]) if score_col and score_col in row and pd.notna(row[score_col]) else None,
                score_label=score_label,
                explanation=row.get(explanation_col) if explanation_col else None,
                poster_url=poster_url,
            )