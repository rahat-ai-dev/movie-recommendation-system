import tempfile
from pathlib import Path

from src.services.tmdb_client import TMDBClient


def test_no_api_key_returns_none_without_crashing():
    with tempfile.TemporaryDirectory() as tmp:
        client = TMDBClient(api_key="", cache_dir=Path(tmp))
        assert client.is_available() is False
        assert client.get_movie_metadata(603) is None
        assert client.get_now_playing() is None


def test_cache_roundtrip_without_live_call():
    with tempfile.TemporaryDirectory() as tmp:
        client = TMDBClient(api_key="", cache_dir=Path(tmp))
        client._write_cache("movie_603", {"title": "The Matrix", "poster_url": None})
        cached = client._read_cache("movie_603")
        assert cached["title"] == "The Matrix"
        # get_movie_metadata should serve this cache even with no live API key
        served = client.get_movie_metadata(603)
        assert served["title"] == "The Matrix"


def test_poster_and_backdrop_url_construction():
    client = TMDBClient(api_key="dummy")
    assert client.poster_url(None) is None
    assert client.poster_url("/abc.jpg").endswith("/abc.jpg")
    assert "w342" in client.poster_url("/abc.jpg")
    assert "w780" in client.backdrop_url("/abc.jpg")
