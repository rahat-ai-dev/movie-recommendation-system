"""
Step 5 of the spec: inspect both datasets BEFORE building the modeling
layer, and produce a documented, real (never-guessed) dataset summary.

Run:  python -m src.data.validator
Writes: docs/dataset_summary.json  (used by the Home page + README numbers)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import config
from src.data import loader


def validate_schema(ratings: pd.DataFrame, movies: pd.DataFrame,
                     tags: pd.DataFrame, links: pd.DataFrame) -> dict:
    issues = []

    expected = {
        "ratings": {"userId", "movieId", "rating", "timestamp"},
        "movies": {"movieId", "title", "genres"},
        "tags": {"userId", "movieId", "tag", "timestamp"},
        "links": {"movieId", "imdbId", "tmdbId"},
    }
    frames = {"ratings": ratings, "movies": movies, "tags": tags, "links": links}
    for name, cols in expected.items():
        missing = cols - set(frames[name].columns)
        if missing:
            issues.append(f"{name}.csv missing columns: {missing}")

    if not ratings["rating"].between(0.5, 5.0).all():
        issues.append("ratings.rating outside expected [0.5, 5.0] range")

    dup_ratings = ratings.duplicated(subset=["userId", "movieId"]).sum()
    dup_movies = movies["movieId"].duplicated().sum()

    return {
        "schema_issues": issues,
        "duplicate_user_movie_ratings": int(dup_ratings),
        "duplicate_movie_ids": int(dup_movies),
        "null_counts": {
            "ratings": ratings.isnull().sum().to_dict(),
            "movies": movies.isnull().sum().to_dict(),
            "tags": tags.isnull().sum().to_dict(),
            "links": links.isnull().sum().to_dict(),
        },
    }


def summarize(dataset_name: str | None = None) -> dict:
    """Build a real, measured summary for the configured dataset (ml-32m).

    `dataset_name` is accepted for call-site compatibility but is currently
    a no-op since ml-32m is the only supported dataset.
    """
    ratings = loader.load_ratings()
    movies = loader.load_movies()
    tags = loader.load_tags()
    links = loader.load_links()

    validation = validate_schema(ratings, movies, tags, links)

    summary = {
        "dataset": config.DATASET,
        "release_note": config.DATASET_RELEASE_DATES.get(config.DATASET, "unknown"),
        "files": {
            "ratings.csv": {"rows": len(ratings), "columns": list(ratings.columns[:4])},
            "movies.csv": {"rows": len(movies), "columns": ["movieId", "title", "genres"]},
            "tags.csv": {"rows": len(tags), "columns": list(tags.columns)},
            "links.csv": {"rows": len(links), "columns": list(links.columns)},
        },
        "unique_users": int(ratings["userId"].nunique()),
        "unique_movies_rated": int(ratings["movieId"].nunique()),
        "unique_movies_catalog": int(movies["movieId"].nunique()),
        "num_ratings": int(len(ratings)),
        "num_tags": int(len(tags)),
        "rating_min": float(ratings["rating"].min()),
        "rating_max": float(ratings["rating"].max()),
        "rating_mean": float(ratings["rating"].mean()),
        "timestamp_min": ratings["ts"].min().strftime("%Y-%m-%d"),
        "timestamp_max": ratings["ts"].max().strftime("%Y-%m-%d"),
        "sparsity_pct": round(
            100 * (1 - len(ratings) / (ratings["userId"].nunique() * movies["movieId"].nunique())),
            4,
        ),
        "validation": validation,
    }

    return summary


if __name__ == "__main__":
    out = {}
    raw_dir = config.DATA_RAW_DIR / config.DATASET
    if raw_dir.exists() and (raw_dir / "ratings.csv").exists():
        out[config.DATASET] = summarize(config.DATASET)
    else:
        print(f"data/raw/{config.DATASET}/ not found -- run "
              f"`python scripts/download_dataset.py` first.")

    out_path = config.PROJECT_ROOT / "docs" / "dataset_summary.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
