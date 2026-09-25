"""
Memory-efficient loaders for the ml-32m MovieLens files (32M ratings).

Uses:
  - explicit, narrow dtypes (int32 ids instead of int64, float32 ratings)
  - a chunked reader for ratings.csv (`iter_ratings_chunks`) for streaming
    aggregate stats without holding all 32M rows in memory at once
"""
from __future__ import annotations

import pandas as pd

from src import config

RATINGS_DTYPES = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
}
TAGS_DTYPES = {
    "userId": "int32",
    "movieId": "int32",
}


def load_ratings(nrows: int | None = None) -> pd.DataFrame:
    """Load ratings.csv with memory-efficient dtypes.

    timestamp is parsed to a proper datetime column `ts` in addition to the
    raw unix `timestamp`, since several stages (time-aware split, EDA) need it.
    """
    df = pd.read_csv(
        config.RATINGS_CSV,
        dtype=RATINGS_DTYPES,
        nrows=nrows,
    )
    df["timestamp"] = df["timestamp"].astype("int64")
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def iter_ratings_chunks(chunksize: int = 2_000_000):
    """Chunked reader for ml-32m -- use when a full in-memory load isn't
    necessary (e.g. streaming aggregate stats)."""
    for chunk in pd.read_csv(
        config.RATINGS_CSV, dtype=RATINGS_DTYPES, chunksize=chunksize
    ):
        chunk["ts"] = pd.to_datetime(chunk["timestamp"].astype("int64"), unit="s")
        yield chunk


def load_movies() -> pd.DataFrame:
    df = pd.read_csv(config.MOVIES_CSV, dtype={"movieId": "int32"})
    # MovieLens titles embed the release year as "Title (YYYY)".
    year = df["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    df["year"] = pd.to_numeric(year, errors="coerce").astype("Int64")
    df["clean_title"] = df["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    df["genres_list"] = df["genres"].apply(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )
    return df


def load_tags() -> pd.DataFrame:
    return pd.read_csv(config.TAGS_CSV, dtype=TAGS_DTYPES, parse_dates=False)


def load_links() -> pd.DataFrame:
    return pd.read_csv(
        config.LINKS_CSV,
        dtype={"movieId": "int32"},
    )


def load_all() -> dict[str, pd.DataFrame]:
    return {
        "ratings": load_ratings(),
        "movies": load_movies(),
        "tags": load_tags(),
        "links": load_links(),
    }
