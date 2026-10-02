"""
TMDB integration layer (spec sections 17-19).

Design:
  TMDB available    -> fetch -> cache to disk (JSON) -> return
  TMDB unavailable   -> serve from disk cache if present
  No cache, no TMDB  -> return None (caller renders a fallback card --
                        NEVER a fabricated poster/rating)

No credentials are hard-coded. Reads TMDB_API_KEY from env / .env /
st.secrets (app layer wires that in). Without a key, `is_available()`
returns False immediately and every call degrades to cache-or-None --
the app must keep working either way (verified in tests/test_tmdb_client.py
by simulating a missing key).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

from src import config

logger = logging.getLogger("tmdb_client")


class TMDBClient:
    def __init__(self, api_key: str | None = None, cache_dir: Path = config.TMDB_CACHE_DIR,
                 timeout: int = config.TMDB_TIMEOUT_SECONDS, max_retries: int = config.TMDB_MAX_RETRIES):
        self.api_key = api_key if api_key is not None else config.TMDB_API_KEY
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> dict | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
            age_days = (time.time() - payload.get("_cached_at", 0)) / 86400
            if age_days > config.TMDB_CACHE_TTL_DAYS:
                logger.info("cache stale for %s (%.1f days old) -- will refresh if TMDB reachable", key, age_days)
            return payload.get("data")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("corrupt cache for %s: %s", key, e)
            return None

    def _write_cache(self, key: str, data: dict) -> None:
        try:
            self._cache_path(key).write_text(json.dumps({"_cached_at": time.time(), "data": data}))
        except OSError as e:
            logger.warning("failed to write cache for %s: %s", key, e)

    def _get(self, endpoint: str, params: dict | None = None) -> dict | None:
        if not self.is_available():
            return None
        params = dict(params or {})
        params["api_key"] = self.api_key

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(f"{config.TMDB_BASE_URL}{endpoint}", params=params, timeout=self.timeout)
                if resp.status_code == 429:  # rate limited
                    wait = float(resp.headers.get("Retry-After", 1))
                    logger.warning("TMDB rate-limited, waiting %.1fs", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_error = e
                logger.warning("TMDB request failed (attempt %d/%d): %s", attempt + 1, self.max_retries + 1, e)
                time.sleep(0.5 * (attempt + 1))
        logger.error("TMDB unreachable after retries: %s", last_error)
        return None

    def get_movie_metadata(self, tmdb_id: int) -> dict | None:
        """Fetch (or serve cached) TMDB metadata for one movie.
        Returns None if neither live TMDB nor a cached copy is available --
        caller must render a graceful fallback, never invented data."""
        cache_key = f"movie_{tmdb_id}"
        cached = self._read_cache(cache_key)

        if not self.is_available():
            return cached  # offline mode: cache or nothing, never fabricated

        raw = self._get(f"/movie/{tmdb_id}")
        if raw is None:
            return cached  # TMDB down -> fall back to cache

        parsed = {
            "tmdb_id": tmdb_id,
            "title": raw.get("title"),
            "overview": raw.get("overview"),
            "release_date": raw.get("release_date"),
            "vote_average": raw.get("vote_average"),
            "vote_count": raw.get("vote_count"),
            "runtime": raw.get("runtime"),
            "genres": [g["name"] for g in raw.get("genres", [])],
            "poster_url": self.poster_url(raw.get("poster_path")),
            "backdrop_url": self.backdrop_url(raw.get("backdrop_path")),
        }
        self._write_cache(cache_key, parsed)
        return parsed

    def poster_url(self, poster_path: str | None) -> str | None:
        if not poster_path:
            return None
        return f"{config.TMDB_IMAGE_BASE}/{config.TMDB_POSTER_SIZE}{poster_path}"

    def backdrop_url(self, backdrop_path: str | None) -> str | None:
        if not backdrop_path:
            return None
        return f"{config.TMDB_IMAGE_BASE}/{config.TMDB_BACKDROP_SIZE}{backdrop_path}"

    def get_now_playing(self, page: int = 1) -> list[dict] | None:
        """Current/recent releases -- explicitly for the 'Discover' current-
        movies experience, kept separate from historical MovieLens data
        (spec section 20)."""
        cache_key = f"now_playing_page{page}"
        cached = self._read_cache(cache_key)
        if not self.is_available():
            return cached
        raw = self._get("/movie/now_playing", {"page": page})
        if raw is None:
            return cached
        results = raw.get("results", [])
        self._write_cache(cache_key, results)
        return results
