# src/features/weather.py
"""Venue weather from Open-Meteo and match start times from Betfair.

    python -m src.features.weather             geocode venues only (review data/raw/weather/venues.csv)
    python -m src.features.weather --download  then download hourly weather (cached, resumable)

Weather values are historical observations, standing in for the forecast a
bettor has at the toss. Requests stay inside Open-Meteo's free tier: <= 14 days
and <= 10 variables each (one call apiece), at most one per second.
"""
from __future__ import annotations

import os
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

from src.config import ROOT, TICKS
from src.features.outcomes import mens_t20_matches, venue_key

WEATHER_DIR = ROOT / "data" / "raw" / "weather"
CACHE = WEATHER_DIR / "cache"
VENUES_CSV = WEATHER_DIR / "venues.csv"
HOURLY = WEATHER_DIR / "hourly.parquet"
MARKETS = TICKS / "match_odds_markets.parquet"
OVERRIDES = ROOT / "src" / "features" / "venue_overrides.csv"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "cloud_cover",
        "wind_speed_10m", "precipitation"]
WEATHER = VARS + ["start_hour_local", "is_night"]
NIGHT_HOUR = 17
DAY_HOURS = (14, 20)
PAUSE_S = 1.0
MAX_RETRIES = 5
BACKOFF_S = 60.0


def venue_cities(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.assign(venue_key=venue_key(matches.venue, matches.city))
    n = m.groupby("venue_key").size().rename("n_matches")
    top = lambda s: s.value_counts().index[0]  # noqa: E731
    city = m.dropna(subset=["city"]).groupby("venue_key").city.agg(top)
    # no city anywhere: the text after the first comma in the raw venue ("X Stadium, Guwahati")
    rest = m[~m.venue_key.isin(city.index)]
    tail = rest.assign(t=rest.venue.str.split(",", n=1).str[1].str.strip())
    tail = tail[tail.t.notna() & (tail.t != "")]
    city = pd.concat([city, tail.groupby("venue_key").t.agg(top)]) if len(tail) else city
    out = pd.DataFrame(n).join(city.rename("city"))
    fallback = pd.Series(out.index.str.split("|").str[0], index=out.index)
    out["city"] = out.city.fillna(fallback)
    return out.sort_values("n_matches", ascending=False)


def venue_countries(matches: pd.DataFrame) -> pd.Series:
    """Most common Betfair `country_code` per venue_key; venues with none are absent.

    NB Betfair labels most non-UK cricket 'GB' (its home market), so GB is only used for
    a venue when no other code was ever seen for it.
    """
    mk = pd.read_parquet(MARKETS, columns=["match_id", "country_code"]).dropna().drop_duplicates()
    m = matches[["match_id", "venue", "city"]].merge(mk, on="match_id")
    m["venue_key"] = venue_key(m.venue, m.city)
    n = m.groupby(["venue_key", "country_code"]).size().rename("n").reset_index()
    n["gb"] = n.country_code == "GB"
    best = n.sort_values(["venue_key", "gb", "n"], ascending=[True, True, False]).drop_duplicates("venue_key")
    return best.set_index("venue_key").country_code


def _request(get, url: str, params: dict, timeout: int):
    """GET with retries on 429, 5xx, connection errors and timeouts."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = get(url, params=params, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout):
            r = None
        if r is not None and r.status_code != 429 and r.status_code < 500:
            r.raise_for_status()
            return r
        if attempt < MAX_RETRIES:
            time.sleep(BACKOFF_S * attempt)
    raise RuntimeError(f"{url} kept failing for {params}")


def geocode(name: str, country_code: str | None = None, get=requests.get) -> dict | None:
    """Top geocoding hit; with `country_code`, the first hit in that country or None."""
    r = _request(get, GEOCODE_URL, {"name": name, "count": 10, "language": "en", "format": "json"}, 30)
    res = r.json().get("results") or []
    if country_code:
        res = [x for x in res if str(x.get("country_code", "")).upper() == country_code.upper()]
    if not res:
        return None
    top = res[0]
    return {"latitude": top["latitude"], "longitude": top["longitude"], "timezone": top.get("timezone"),
            "country": top.get("country"), "country_code": top.get("country_code")}


def _geocode_venue(city: str, name: str, cc, get, pause: float):
    """City then venue name; (hit, source). A GB code is Betfair's default, so when it finds
    nothing the lookup is repeated without a country and the source is marked for review."""
    for code, tag in ((cc, ""), (None, "+unverified_country")) if cc == "GB" else ((cc, ""),):
        for q, src in ((city, "geocode:city"), *(((name, "geocode:venue"),) if name != city else ())):
            g = geocode(q, code, get)
            time.sleep(pause)
            if g:
                return g, src + tag
    return None, "unresolved"


def _same(a, b) -> bool:
    return (pd.isna(a) and pd.isna(b)) or a == b


def build_venues(matches: pd.DataFrame, get=requests.get, pause: float = PAUSE_S,
                 countries: pd.Series | None = None) -> pd.DataFrame:
    vc = venue_cities(matches)
    countries = venue_countries(matches) if countries is None else countries
    over = (pd.read_csv(OVERRIDES).set_index("venue_key") if OVERRIDES.exists()
            else pd.DataFrame(columns=["latitude", "longitude", "timezone", "country"]))
    old = pd.read_csv(VENUES_CSV).set_index("venue_key") if VENUES_CSV.exists() else None
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for vk, r in vc.iterrows():
        cc = countries.get(vk)
        rec = {"venue_key": vk, "city": r.city, "n_matches": int(r.n_matches), "country_code": cc}
        if vk in over.index:
            o = over.loc[vk].to_dict()
            if any(pd.isna(o.get(k)) for k in ("latitude", "longitude", "timezone")):
                raise ValueError(f"venue_overrides.csv: {vk!r} needs latitude, longitude and timezone")
            rec.update(o, source="override")
            rec["country_code"] = o.get("country_code", cc)
        elif (old is not None and vk in old.index and pd.notna(old.loc[vk, "latitude"])
              and old.loc[vk, "source"] != "override" and _same(old.loc[vk].get("country_code"), cc)):
            rec.update(old.loc[vk, ["latitude", "longitude", "timezone", "country", "source"]].to_dict())
        else:
            g, src = _geocode_venue(r.city, vk.split("|")[0], cc, get, pause)
            rec.update(g or {}, source=src)
            rec["country_code"] = cc          # the code we asked for; geocode's own is in g
        rows.append(rec)
        # save after every venue; venues not reached yet keep their previous rows
        done = pd.DataFrame(rows).set_index("venue_key")
        keep = old[~old.index.isin(done.index)] if old is not None else None
        (pd.concat([done, keep]) if keep is not None else done).to_csv(VENUES_CSV)
    return pd.DataFrame(rows).set_index("venue_key")


def date_windows(dates, max_days: int = 14):
    ds = sorted(set(pd.to_datetime(pd.Series(list(dates))).dt.date))
    out, start, end = [], None, None
    for d in ds:
        if start is None:
            start = end = d
        elif (d - start).days < max_days:
            end = d
        else:
            out.append((start, end))
            start = end = d
    if start is not None:
        out.append((start, end))
    return out


def fetch_window(lat, lon, start, end, get=requests.get) -> pd.DataFrame:
    """Hourly rows in UTC (`timezone=GMT`), so daylight saving never shifts the grid."""
    params = {"latitude": lat, "longitude": lon, "start_date": str(start), "end_date": str(end),
              "hourly": ",".join(VARS), "timezone": "GMT"}
    p = _request(get, ARCHIVE_URL, params, 60).json()
    h = pd.DataFrame(p["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    h["utc_offset_seconds"] = int(p.get("utc_offset_seconds", 0))
    return h[["time", "utc_offset_seconds", *VARS]]


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def download_weather(venues: pd.DataFrame, matches: pd.DataFrame, get=requests.get,
                     pause: float = PAUSE_S) -> pd.DataFrame:
    """`venues` indexed by venue_key with latitude/longitude; `matches` has venue_key, date.

    A cache file is named by venue, coordinates and window, so moving a venue refetches it.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    have = venues.dropna(subset=["latitude", "longitude"])
    used = []
    for vk, v in have.iterrows():
        for start, end in date_windows(matches.loc[matches.venue_key == vk, "date"]):
            path = CACHE / f"{_slug(vk)}_{v.latitude:.3f}_{v.longitude:.3f}_{start}_{end}.parquet"
            used.append(path)
            if path.exists():
                continue
            h = fetch_window(v.latitude, v.longitude, start, end, get).assign(venue_key=vk)
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            h.to_parquet(tmp)
            os.replace(tmp, path)
            time.sleep(pause)
    parts = [pd.read_parquet(p) for p in used]
    hourly = (pd.concat(parts, ignore_index=True).drop_duplicates(["venue_key", "time"])
              if parts else pd.DataFrame(columns=["venue_key", "time", "utc_offset_seconds", *VARS]))
    hourly.to_parquet(HOURLY)
    return hourly


def start_times() -> pd.Series:
    mo = pd.read_parquet(TICKS / "match_odds_markets.parquet", columns=["match_id", "market_time"])
    mo["market_time"] = pd.to_datetime(mo.market_time, utc=True)
    mo = mo.dropna(subset=["match_id", "market_time"])
    mo = mo[mo.market_time.dt.year < 2030]          # 2099 placeholder start times
    return mo.groupby("match_id").market_time.min()


def _utc(t: pd.Series) -> pd.Series:
    return pd.to_datetime(t, utc=True).dt.as_unit("ns")


def match_weather(mv: pd.DataFrame, hourly: pd.DataFrame, starts: pd.Series,
                  tz: pd.Series) -> pd.DataFrame:
    """`hourly.time` is UTC; `tz` maps venue_key to its IANA zone. Venues without a zone, or
    without hourly rows, come out all NaN; matches without a start get NaN start fields and
    the 14:00-20:59 local-time window of the match date instead of the start hour."""
    assert not hourly.duplicated(["venue_key", "time"]).any(), "duplicate (venue_key, time) rows"
    m = mv[["match_id", "date", "venue_key"]].copy().reset_index(drop=True)
    m["day"] = pd.to_datetime(m.date).dt.normalize().dt.as_unit("ns")
    m["tz"] = m.venue_key.map(tz)
    h = hourly.assign(time=_utc(hourly.time))
    start = _utc(pd.Series(starts.reindex(m.match_id).to_numpy(), index=m.index))
    m["start_hour_local"] = np.nan
    for name, idx in m.groupby("tz").groups.items():
        loc = start[idx].dt.tz_convert(name)
        m.loc[idx, "start_hour_local"] = loc.dt.hour + loc.dt.minute / 60
    m["is_night"] = (m.start_hour_local >= NIGHT_HOUR).astype(float).where(m.start_hour_local.notna())
    m["time"] = start.dt.floor("h")
    at = m[["venue_key", "time"]].merge(h[["venue_key", "time", *VARS]],
                                        on=["venue_key", "time"], how="left")[VARS]
    # day window in the venue's local clock
    h["tz"] = h.venue_key.map(tz)
    local = [g[["venue_key", *VARS]].assign(day=(t := g.time.dt.tz_convert(name)).dt.tz_localize(None)
                                            .dt.normalize().dt.as_unit("ns"), hour=t.dt.hour)
             for name, g in h.groupby("tz")]
    hl = pd.concat(local, ignore_index=True) if local else pd.DataFrame(
        {"venue_key": pd.Series(dtype=object), "day": pd.Series(dtype="datetime64[ns]"),
         "hour": pd.Series(dtype=int), **{v: pd.Series(dtype=float) for v in VARS}})
    win = hl[hl.hour.between(*DAY_HOURS)].groupby(["venue_key", "day"])[VARS].mean().reset_index()
    day = m[["venue_key", "day"]].merge(win, on=["venue_key", "day"], how="left")[VARS]
    have = at.notna().all(axis=1).to_numpy()
    vals = at.where(np.repeat(have[:, None], len(VARS), axis=1), day)
    vals[m.tz.isna().to_numpy()] = np.nan
    out = pd.concat([vals, m[["start_hour_local", "is_night"]]], axis=1)
    out.index = m.match_id
    return out[WEATHER].astype(float)


def load_match_weather(balls: pd.DataFrame) -> pd.DataFrame:
    for f in (HOURLY, VENUES_CSV):
        if not f.exists():
            raise FileNotFoundError(f"{f} missing: run `python -m src.features.weather --download` first")
    mv = balls[["match_id", "date", "venue_key"]].drop_duplicates("match_id")
    tz = pd.read_csv(VENUES_CSV).set_index("venue_key").timezone
    return match_weather(mv, pd.read_parquet(HOURLY), start_times(), tz)


if __name__ == "__main__":
    matches = mens_t20_matches()
    venues = build_venues(matches)
    print(f"{len(venues)} venues")
    print("unresolved:", venues[venues.latitude.isna()].index.tolist())
    print("no country_code:", venues[venues.country_code.isna()].index.tolist())
    print("country unverified:", venues[venues.source.str.contains("unverified", na=False)].index.tolist())
    print(venues.head(60)[["city", "country", "country_code", "latitude", "longitude", "source",
                           "n_matches"]].to_string())
    if "--download" not in sys.argv:
        raise SystemExit(f"review {VENUES_CSV}, then rerun with --download")
    mv = matches.assign(venue_key=venue_key(matches.venue, matches.city))[["venue_key", "date"]]
    hourly = download_weather(venues, mv)
    print(f"hourly rows: {len(hourly):,}")
