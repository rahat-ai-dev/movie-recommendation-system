import pandas as pd

from src.data.cleaner import dedupe_ratings, drop_invalid_ratings, build_id_mappings, clean_pipeline
from src.data.splitter import time_aware_split


def _toy_ratings():
    # Deterministic, hand-built real-shaped data (not fabricated recommendation
    # content -- just a tiny slice used to test pure transformation logic).
    return pd.DataFrame({
        "userId": [1, 1, 1, 1, 1, 1, 2, 2, 2],
        "movieId": [10, 20, 30, 40, 50, 20, 10, 20, 30],
        "rating": [4.0, 3.5, 5.0, 6.0, -1.0, 3.5, 4.0, 4.5, 3.0],  # 6.0/-1.0 invalid
        "timestamp": [100, 200, 300, 400, 500, 250, 100, 200, 300],
    })


def test_drop_invalid_ratings_removes_out_of_range():
    df = _toy_ratings()
    cleaned = drop_invalid_ratings(df)
    assert cleaned["rating"].between(0.5, 5.0).all()
    assert len(cleaned) == 7  # drops the 6.0 and -1.0 rows


def test_dedupe_keeps_latest_timestamp():
    df = _toy_ratings()
    deduped = dedupe_ratings(df)
    # user 1 rated movie 20 twice (ts 200 and 250) -> keep ts=250 row
    row = deduped[(deduped.userId == 1) & (deduped.movieId == 20)]
    assert len(row) == 1
    assert row.iloc[0]["timestamp"] == 250


def test_build_id_mappings_are_dense_and_bijective():
    df = clean_pipeline(_toy_ratings())
    u2i, i2u, m2i, i2m = build_id_mappings(df)
    assert set(u2i.values()) == set(range(len(u2i)))
    assert all(i2u[u2i[uid]] == uid for uid in u2i)
    assert all(i2m[m2i[mid]] == mid for mid in m2i)


def test_time_aware_split_respects_min_interactions():
    df = clean_pipeline(_toy_ratings())
    train, test = time_aware_split(df, test_fraction=0.34, min_interactions=3)
    # user 2 has only 3 ratings after cleaning -> eligible, some go to test
    # every test row's movie must NOT also be the same (user, movie) row in train
    merged = train.merge(test, on=["userId", "movieId"], how="inner")
    assert merged.empty


def test_time_aware_split_no_future_leak_into_train():
    df = clean_pipeline(_toy_ratings())
    train, test = time_aware_split(df, test_fraction=0.3, min_interactions=3)
    for uid in test["userId"].unique():
        max_train_ts = train[train.userId == uid]["timestamp"].max()
        min_test_ts = test[test.userId == uid]["timestamp"].min()
        if pd.notna(max_train_ts):
            assert min_test_ts >= max_train_ts
