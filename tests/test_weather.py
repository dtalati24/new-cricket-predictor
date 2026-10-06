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


def test_geocode_parses_the_top_result_and_respects_the_country():
    res = [{"latitude": 43.2, "longitude": -79.9, "timezone": "America/Toronto", "country": "Canada",
            "country_code": "CA"},
           {"latitude": -37.8, "longitude": 175.3, "timezone": "Pacific/Auckland",
            "country": "New Zealand", "country_code": "NZ"}]
    seen = []

    def get(url, params, timeout):
        seen.append(params["count"])
        return FakeResponse({"results": res})

    assert W.geocode("Hamilton", get=get)["country_code"] == "CA", "no code: first result"
    nz = W.geocode("Hamilton", "nz", get=get)
    assert nz == {"latitude": -37.8, "longitude": 175.3, "timezone": "Pacific/Auckland",
                  "country": "New Zealand", "country_code": "NZ"}
    assert W.geocode("Hamilton", "BM", get=get) is None, "never a wrong-country hit"
    assert seen == [10, 10, 10]
    assert W.geocode("Nowhere", get=lambda u, params, timeout: FakeResponse({})) is None


def test_requests_retry_on_errors_and_429(monkeypatch):
    import requests
    monkeypatch.setattr(W, "BACKOFF_S", 0)
    answers = iter([requests.ConnectionError(), FakeResponse({}, 429), requests.Timeout(),
                    FakeResponse({"results": [{"latitude": 1, "longitude": 2, "country_code": "IN"}]})])

    def get(url, params, timeout):
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a

    assert W.geocode("X", get=get)["latitude"] == 1
    with pytest.raises(RuntimeError):
        W.geocode("X", get=lambda u, params, timeout: FakeResponse({}, 503))


def _matches():
    return pd.DataFrame({
        "match_id": ["1", "2", "3", "4", "5"],
        "venue": ["Seddon Park", "Seddon Park", "Seddon Park", "Barsapara Cricket Stadium, Guwahati",
                  "Motera Stadium"],
        "city": ["Hamilton", "Hamilton", None, None, None]})


def test_venue_cities_uses_the_text_after_the_comma_when_there_is_no_city():
    vc = W.venue_cities(_matches())
    assert vc.loc["Seddon Park", "city"] == "Hamilton" and vc.loc["Seddon Park", "n_matches"] == 3
    assert vc.loc["Barsapara Cricket Stadium", "city"] == "Guwahati"
    assert vc.loc["Motera Stadium", "city"] == "Motera Stadium", "falls back to the venue name"


def test_venue_countries_is_the_commonest_code_and_gb_is_only_a_default(tmp_path, monkeypatch):
    mk = pd.DataFrame({"match_id": ["1", "2", "3", "3", "4", "5"],
                       "country_code": ["NZ", "NZ", "GB", "NZ", "GB", None]})
    mk.to_parquet(tmp_path / "mk.parquet")
    monkeypatch.setattr(W, "MARKETS", tmp_path / "mk.parquet")
    c = W.venue_countries(_matches())
    assert c.to_dict() == {"Seddon Park": "NZ", "Barsapara Cricket Stadium": "GB"}


def test_build_venues_passes_the_country_saves_each_venue_and_validates_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "WEATHER_DIR", tmp_path)
    monkeypatch.setattr(W, "VENUES_CSV", tmp_path / "venues.csv")
    monkeypatch.setattr(W, "OVERRIDES", tmp_path / "over.csv")
    (tmp_path / "over.csv").write_text("venue_key,latitude,longitude,timezone,country\n")
    asked = []

    def get(url, params, timeout):
        asked.append(params["name"])
        hit = {"latitude": 1.0, "longitude": 2.0, "timezone": "Z/Z", "country": "C", "country_code": "NZ"}
        if params["name"] == "Guwahati":
            return FakeResponse({"results": [hit]})
        if params["name"] == "Hamilton":
            return FakeResponse({"results": [dict(hit, country_code="CA")]})
        return FakeResponse({})

    countries = pd.Series({"Seddon Park": "NZ", "Barsapara Cricket Stadium": "GB"})
    out = W.build_venues(_matches(), get=get, pause=0, countries=countries)
    assert out.loc["Seddon Park", "source"] == "unresolved", "the Hamilton hit is in CA, not NZ"
    assert asked.index("Hamilton") < asked.index("Seddon Park"), "city is tried before the venue name"
    bar = out.loc["Barsapara Cricket Stadium"]
    assert bar.source == "geocode:city+unverified_country" and bar.country_code == "GB"
    saved = pd.read_csv(W.VENUES_CSV).set_index("venue_key")
    assert saved.loc["Barsapara Cricket Stadium", "latitude"] == 1.0
    n = len(asked)
    W.build_venues(_matches(), get=get, pause=0, countries=countries)
    assert "Guwahati" not in asked[n:], "resolved venues are reused"
    (tmp_path / "over.csv").write_text("venue_key,latitude,longitude,timezone,country\nSeddon Park,1,2,,NZ\n")
    with pytest.raises(ValueError, match="Seddon Park"):
        W.build_venues(_matches(), get=get, pause=0, countries=countries)


def test_a_gb_code_falls_back_to_an_unverified_country_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "WEATHER_DIR", tmp_path)
    monkeypatch.setattr(W, "VENUES_CSV", tmp_path / "venues.csv")
    monkeypatch.setattr(W, "OVERRIDES", tmp_path / "none.csv")
    get = lambda u, params, timeout: FakeResponse({"results": [  # noqa: E731
        {"latitude": 1.0, "longitude": 2.0, "timezone": "Asia/Kolkata", "country": "India",
         "country_code": "IN"}]})
    out = W.build_venues(_matches().iloc[[3]], get=get, pause=0,
                         countries=pd.Series({"Barsapara Cricket Stadium": "GB"}))
    r = out.loc["Barsapara Cricket Stadium"]
    assert r.source == "geocode:city+unverified_country" and r.country == "India"


def _hits(*codes):
    return {c: {"latitude": 1.0, "longitude": 2.0, "timezone": "Z/Z", "country": c, "country_code": c}
            for c in codes}


def _search(hits_by_name):
    """fake get: results for a name, in the given order of country codes"""
    def get(url, params, timeout):
        return FakeResponse({"results": [_hits(c)[c] for c in hits_by_name.get(params["name"], [])]})
    return get


def test_weak_codes_are_hints_that_get_cross_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "WEATHER_DIR", tmp_path)
    monkeypatch.setattr(W, "VENUES_CSV", tmp_path / "venues.csv")
    monkeypatch.setattr(W, "OVERRIDES", tmp_path / "none.csv")
    pg = lambda cc, names: W._geocode_venue("City", "Venue", cc, _search(names), 0)  # noqa: E731
    # weak code, filtered and unfiltered agree: trusted
    assert pg("GB", {"City": ["GB", "IN"]})[1] == "geocode:city"
    # weak code, filtered hit exists but the unfiltered top is another country: keep filtered, tag
    g, src = pg("GB", {"City": ["IN", "GB"]})
    assert g["country_code"] == "GB" and src == "geocode:city+unverified_country"
    # weak code, nothing in that country: unfiltered hit, tagged
    g, src = pg("JM", {"City": ["GY"]})
    assert g["country_code"] == "GY" and src == "geocode:city+unverified_country"
    # strong code never leaves its country
    assert pg("IN", {"City": ["GY"]}) == (None, "unresolved")
    assert pg("IN", {"City": ["GY", "IN"]})[1] == "geocode:city"
    # no code at all: tagged
    assert pg(np.nan, {"Venue": ["IN"]})[1] == "geocode:venue+unverified_country"


def test_venue_countries_treats_gb_and_jm_as_weak(tmp_path, monkeypatch):
    mk = pd.DataFrame({"match_id": ["1", "2", "3", "4", "5"],
                       "country_code": ["JM", "JM", "JM", "GY", "GB"]})
    mk.to_parquet(tmp_path / "mk.parquet")
    monkeypatch.setattr(W, "MARKETS", tmp_path / "mk.parquet")
    m = pd.DataFrame({"match_id": ["1", "2", "3", "4", "5"], "venue": ["P", "P", "P", "P", "Q"],
                      "city": ["x"] * 5})
    assert W.venue_countries(m).to_dict() == {"P": "GY", "Q": "GB"}


def test_check_venues_flags_a_wrong_timezone():
    venues = pd.DataFrame({"timezone": ["Asia/Kolkata", "Europe/London", None],
                           "n_matches": [5, 3, 9]},
                          index=pd.Index(["Mumbai Ground", "Wrong Ground", "No Tz"], name="venue_key"))
    matches = pd.DataFrame({"match_id": ["a", "b", "c", "d", "e"],
                            "venue": ["Mumbai Ground"] * 2 + ["Wrong Ground"] * 2 + ["No Tz"],
                            "city": None})
    # Mumbai: 14:00Z = 19:30 local; Wrong Ground: 02:00Z = 02:00 (GMT in January); No Tz: skipped
    starts = pd.Series(pd.to_datetime(["2024-01-10 14:00", "2024-01-11 14:00", "2024-01-10 02:00",
                                       "2024-01-11 02:00", "2024-01-10 02:00"], utc=True),
                       index=list("abcde"))
    out = W.check_venues(venues, matches, starts)
    assert out.index.tolist() == ["Mumbai Ground", "Wrong Ground"]
    assert out.loc["Mumbai Ground", "median_hour"] == 19.5 and not out.loc["Mumbai Ground", "suspect"]
    assert out.loc["Wrong Ground", "median_hour"] == 2.0 and out.loc["Wrong Ground", "suspect"]
    assert out.loc["Wrong Ground", "n_start"] == 2


def test_date_windows_never_exceed_14_days():
    d = [dt.date(2024, 1, 1), dt.date(2024, 1, 5), dt.date(2024, 1, 14), dt.date(2024, 1, 15),
         dt.date(2024, 3, 1)]
    w = W.date_windows(d)
    assert w == [(dt.date(2024, 1, 1), dt.date(2024, 1, 14)),
                 (dt.date(2024, 1, 15), dt.date(2024, 1, 15)),
                 (dt.date(2024, 3, 1), dt.date(2024, 3, 1))]


def test_download_windows_cover_the_day_before_and_after_each_match():
    for dates, want in ([["2024-05-02"], [("2024-05-01", "2024-05-03")]],
                        [["2024-05-02", "2024-05-20", "2024-05-21"],
                         [("2024-05-01", "2024-05-03"), ("2024-05-19", "2024-05-22")]]):
        w = [(str(a), str(b)) for a, b in W.date_windows(W._padded(pd.to_datetime(dates)))]
        assert w == want and all((pd.Timestamp(b) - pd.Timestamp(a)).days < 14 for a, b in w)


def _hourly(start="2024-05-02 00:00", periods=24, venue="V"):
    """UTC hourly rows; VARS[i] at row r is r + 100 * i."""
    times = pd.date_range(start, periods=periods, freq="h")
    h = pd.DataFrame({"venue_key": venue, "time": times, "utc_offset_seconds": 0})
    for i, v in enumerate(W.VARS):
        h[v] = np.arange(periods, dtype=float) + 100 * i
    return h


TZ = pd.Series({"V": "Asia/Kolkata", "L": "Europe/London"})


def test_match_weather_uses_the_start_hour_when_known_else_the_day_window():
    mv = pd.DataFrame({"match_id": ["a", "b"], "date": pd.to_datetime(["2024-05-02"] * 2),
                       "venue_key": ["V", "V"]})
    starts = pd.Series(pd.to_datetime(["2024-05-02 14:00"], utc=True), index=["a"])  # 19:30 local
    out = W.match_weather(mv, _hourly(), starts, TZ)
    assert out.loc["a", "temperature_2m"] == 14.0, "start 14:00 UTC is the 14:00 UTC row"
    assert out.loc["a", "start_hour_local"] == pytest.approx(19.5)
    assert out.loc["a", "is_night"] == 1.0
    # local 14:00-20:59 on 2024-05-02 in Kolkata is 08:30-15:29 UTC -> UTC rows 9..15
    assert out.loc["b", "temperature_2m"] == pytest.approx(np.mean(range(9, 16)))
    assert np.isnan(out.loc["b", "start_hour_local"]) and np.isnan(out.loc["b", "is_night"])


def test_daylight_saving_shifts_local_time_and_the_day_window():
    h = _hourly("2024-03-30 00:00", 48, "L")          # BST starts 2024-03-31 01:00 UTC
    mv = pd.DataFrame({"match_id": ["a30", "a31", "b30", "b31"], "venue_key": "L",
                       "date": pd.to_datetime(["2024-03-30", "2024-03-31"] * 2)})
    starts = pd.Series(pd.to_datetime(["2024-03-30 18:00", "2024-03-31 18:00"], utc=True),
                       index=["a30", "a31"])
    out = W.match_weather(mv, h, starts, TZ)
    assert out.loc["a30", "start_hour_local"] == 18.0 and out.loc["a31", "start_hour_local"] == 19.0
    assert out.loc["a30", "temperature_2m"] == 18.0 and out.loc["a31", "temperature_2m"] == 24 + 18
    assert out.loc["b30", "temperature_2m"] == pytest.approx(np.mean(range(14, 21)))          # GMT
    assert out.loc["b31", "temperature_2m"] == pytest.approx(np.mean(range(24 + 13, 24 + 20)))  # BST


def test_venue_without_coordinates_or_timezone_gets_nan_weather():
    mv = pd.DataFrame({"match_id": ["c", "d"], "date": pd.to_datetime(["2024-05-02"] * 2),
                       "venue_key": ["Z", "NOTZ"]})
    out = W.match_weather(mv, pd.concat([_hourly(), _hourly(venue="NOTZ")]),
                          pd.Series(pd.to_datetime(["2024-05-02 14:00"] * 2, utc=True), index=["c", "d"]),
                          TZ)
    assert out.isna().all().all()


def test_match_weather_rejects_duplicate_hours():
    mv = pd.DataFrame({"match_id": ["a"], "date": pd.to_datetime(["2024-05-02"]), "venue_key": ["V"]})
    with pytest.raises(AssertionError):
        W.match_weather(mv, pd.concat([_hourly(), _hourly()]),
                        pd.Series(dtype="datetime64[ns, UTC]"), TZ)


def test_download_skips_cached_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(W, "HOURLY", tmp_path / "hourly.parquet")
    calls = []

    def get(url, params, timeout):
        assert params["timezone"] == "GMT"
        calls.append(params["start_date"])
        n = 24
        return FakeResponse({"utc_offset_seconds": 0, "hourly": {
            "time": [f"{params['start_date']}T{h:02d}:00" for h in range(n)],
            **{v: [1.0] * n for v in W.VARS}}})

    venues = pd.DataFrame({"latitude": [1.0], "longitude": [2.0]}, index=pd.Index(["V"], name="venue_key"))
    matches = pd.DataFrame({"venue_key": ["V"], "date": pd.to_datetime(["2024-05-02"])})
    W.download_weather(venues, matches, get=get, pause=0)
    W.download_weather(venues, matches, get=get, pause=0)
    assert calls == ["2024-05-01"], "second run reuses the cache; window starts a day early"
    assert pd.read_parquet(tmp_path / "hourly.parquet").venue_key.unique().tolist() == ["V"]
    assert [p.name for p in (tmp_path / "cache").iterdir()] == ["V_1.000_2.000_2024-05-01_2024-05-03.parquet"]
    venues.loc["V", "latitude"] = 1.5                                  # corrected coordinates refetch
    out = W.download_weather(venues, matches, get=get, pause=0)
    assert calls == ["2024-05-01"] * 2 and len(out) == 24, "old-coordinate rows are not mixed in"
