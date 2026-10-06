# tests/test_weather.py
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.features import weather as W


class FakeResponse:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(response=self)


def test_geocode_parses_the_top_result():
    get = lambda url, params, timeout: FakeResponse({"results": [  # noqa: E731
        {"latitude": 19.0, "longitude": 72.8, "timezone": "Asia/Kolkata", "country": "India"}]})
    assert W.geocode("Mumbai", get=get) == {"latitude": 19.0, "longitude": 72.8,
                                             "timezone": "Asia/Kolkata", "country": "India"}
    assert W.geocode("Nowhere", get=lambda u, params, timeout: FakeResponse({})) is None


def test_date_windows_never_exceed_14_days():
    d = [dt.date(2024, 1, 1), dt.date(2024, 1, 5), dt.date(2024, 1, 14), dt.date(2024, 1, 15),
         dt.date(2024, 3, 1)]
    w = W.date_windows(d)
    assert w == [(dt.date(2024, 1, 1), dt.date(2024, 1, 14)),
                 (dt.date(2024, 1, 15), dt.date(2024, 1, 15)),
                 (dt.date(2024, 3, 1), dt.date(2024, 3, 1))]


def _hourly():
    times = pd.date_range("2024-05-02 00:00", periods=24, freq="h")
    h = pd.DataFrame({"venue_key": "V", "time": times, "utc_offset_seconds": 19800})
    for i, v in enumerate(W.VARS):
        h[v] = np.arange(24, dtype=float) + 100 * i
    return h


def test_match_weather_uses_the_start_hour_when_known_else_the_day_window():
    mv = pd.DataFrame({"match_id": ["a", "b"], "date": pd.to_datetime(["2024-05-02"] * 2),
                       "venue_key": ["V", "V"]})
    starts = pd.Series(pd.to_datetime(["2024-05-02 14:00"], utc=True), index=["a"])  # 19:30 local
    out = W.match_weather(mv, _hourly(), starts)
    assert out.loc["a", "temperature_2m"] == 19.0, "local 19:30 floors to the 19:00 row"
    assert out.loc["a", "start_hour_local"] == pytest.approx(19.5)
    assert out.loc["a", "is_night"] == 1.0
    assert out.loc["b", "temperature_2m"] == pytest.approx(np.mean(range(14, 21)))
    assert np.isnan(out.loc["b", "start_hour_local"]) and np.isnan(out.loc["b", "is_night"])


def test_venue_without_coordinates_gets_nan_weather():
    mv = pd.DataFrame({"match_id": ["c"], "date": pd.to_datetime(["2024-05-02"]), "venue_key": ["Z"]})
    out = W.match_weather(mv, _hourly(), pd.Series(dtype="datetime64[ns, UTC]"))
    assert out.loc["c"].isna().all()


def test_download_skips_cached_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(W, "HOURLY", tmp_path / "hourly.parquet")
    calls = []

    def get(url, params, timeout):
        calls.append(params["start_date"])
        n = 24
        return FakeResponse({"utc_offset_seconds": 0, "hourly": {
            "time": [f"{params['start_date']}T{h:02d}:00" for h in range(n)],
            **{v: [1.0] * n for v in W.VARS}}})

    venues = pd.DataFrame({"latitude": [1.0], "longitude": [2.0]}, index=pd.Index(["V"], name="venue_key"))
    matches = pd.DataFrame({"venue_key": ["V"], "date": pd.to_datetime(["2024-05-02"])})
    W.download_weather(venues, matches, get=get, pause=0)
    W.download_weather(venues, matches, get=get, pause=0)
    assert calls == ["2024-05-02"], "second run reuses the cache"
    assert pd.read_parquet(tmp_path / "hourly.parquet").venue_key.unique().tolist() == ["V"]
