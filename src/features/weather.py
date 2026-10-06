# src/features/weather.py
"""Venue weather from Open-Meteo and match start times from Betfair.

    python -m src.features.weather      geocode venues, then download (cached, resumable)

Weather values are historical observations, standing in for the forecast a
bettor has at the toss. Requests stay inside Open-Meteo's free tier: <= 14 days
and <= 10 variables each (one call apiece), at most one per second.
"""
from __future__ import annotations

import re
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


def venue_cities(matches: pd.DataFrame) -> pd.DataFrame:
    m = matches.assign(venue_key=venue_key(matches.venue, matches.city))
    n = m.groupby("venue_key").size().rename("n_matches")
    city = m.dropna(subset=["city"]).groupby("venue_key").city.agg(lambda s: s.value_counts().index[0])
    out = pd.DataFrame(n).join(city.rename("city"))
    fallback = pd.Series(out.index.str.split("|").str[0], index=out.index)
    out["city"] = out.city.fillna(fallback)
    return out.sort_values("n_matches", ascending=False)


def geocode(name: str, get=requests.get) -> dict | None:
    r = get(GEOCODE_URL, params={"name": name, "count": 1, "language": "en", "format": "json"},
            timeout=30)
    r.raise_for_status()
    res = r.json().get("results") or []
    if not res:
        return None
    top = res[0]
    return {"latitude": top["latitude"], "longitude": top["longitude"],
            "timezone": top.get("timezone"), "country": top.get("country")}


def build_venues(matches: pd.DataFrame, get=requests.get, pause: float = PAUSE_S) -> pd.DataFrame:
    vc = venue_cities(matches)
    over = (pd.read_csv(OVERRIDES).set_index("venue_key") if OVERRIDES.exists()
            else pd.DataFrame(columns=["latitude", "longitude", "timezone", "country"]))
    old = pd.read_csv(VENUES_CSV).set_index("venue_key") if VENUES_CSV.exists() else None
    rows = []
    for vk, r in vc.iterrows():
        rec = {"venue_key": vk, "city": r.city, "n_matches": int(r.n_matches)}
        if vk in over.index:
            rec.update(over.loc[vk].to_dict(), source="override")
        elif old is not None and vk in old.index and pd.notna(old.loc[vk, "latitude"]) \
                and old.loc[vk, "source"] != "override":
            rec.update(old.loc[vk, ["latitude", "longitude", "timezone", "country", "source"]].to_dict())
        else:
            g, src = geocode(r.city, get), "geocode:city"
            name = vk.split("|")[0]
            if g is None and name != r.city:
                g, src = geocode(name, get), "geocode:venue"
            rec.update(g or {}, source=src if g else "unresolved")
            time.sleep(pause)
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("venue_key")
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(VENUES_CSV)
    return out


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
    params = {"latitude": lat, "longitude": lon, "start_date": str(start), "end_date": str(end),
              "hourly": ",".join(VARS), "timezone": "auto"}
    for attempt in range(1, MAX_RETRIES + 1):
        r = get(ARCHIVE_URL, params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(60 * attempt)
            continue
        r.raise_for_status()
        p = r.json()
        h = pd.DataFrame(p["hourly"])
        h["time"] = pd.to_datetime(h["time"])
        h["utc_offset_seconds"] = int(p.get("utc_offset_seconds", 0))
        return h[["time", "utc_offset_seconds", *VARS]]
    raise RuntimeError(f"Open-Meteo kept failing for {lat},{lon} {start}..{end}")


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def download_weather(venues: pd.DataFrame, matches: pd.DataFrame, get=requests.get,
                     pause: float = PAUSE_S) -> pd.DataFrame:
    """`venues` indexed by venue_key with latitude/longitude; `matches` has venue_key, date."""
    CACHE.mkdir(parents=True, exist_ok=True)
    have = venues.dropna(subset=["latitude", "longitude"])
    for vk, v in have.iterrows():
        for start, end in date_windows(matches.loc[matches.venue_key == vk, "date"]):
            path = CACHE / f"{_slug(vk)}_{start}_{end}.parquet"
            if path.exists():
                continue
            h = fetch_window(v.latitude, v.longitude, start, end, get).assign(venue_key=vk)
            h.to_parquet(path)
            time.sleep(pause)
    parts = [pd.read_parquet(p) for p in sorted(CACHE.glob("*.parquet"))]
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


def match_weather(mv: pd.DataFrame, hourly: pd.DataFrame, starts: pd.Series) -> pd.DataFrame:
    m = mv[["match_id", "date", "venue_key"]].copy().reset_index(drop=True)
    m["day"] = pd.to_datetime(m.date).dt.normalize()
    h = hourly.assign(time=pd.to_datetime(hourly.time))
    h = h.assign(day=h.time.dt.normalize(), hour=h.time.dt.hour)
    off = h.groupby(["venue_key", "day"]).utc_offset_seconds.first().rename("off").reset_index()
    m = m.merge(off, on=["venue_key", "day"], how="left")
    start = pd.Series(starts.reindex(m.match_id).to_numpy(), index=m.index)
    start = pd.to_datetime(start, utc=True).dt.tz_localize(None)
    local = start + pd.to_timedelta(m.off, unit="s")
    m["start_hour_local"] = local.dt.hour + local.dt.minute / 60
    m["is_night"] = (m.start_hour_local >= NIGHT_HOUR).astype(float).where(m.start_hour_local.notna())
    m["time"] = local.dt.floor("h")
    at = m[["venue_key", "time"]].merge(h[["venue_key", "time", *VARS]],
                                        on=["venue_key", "time"], how="left")[VARS]
    win = h[h.hour.between(*DAY_HOURS)].groupby(["venue_key", "day"])[VARS].mean().reset_index()
    day = m[["venue_key", "day"]].merge(win, on=["venue_key", "day"], how="left")[VARS]
    have = at.notna().all(axis=1).to_numpy()
    vals = at.where(np.repeat(have[:, None], len(VARS), axis=1), day)
    out = pd.concat([vals, m[["start_hour_local", "is_night"]]], axis=1)
    out.index = m.match_id
    return out[WEATHER].astype(float)


def load_match_weather(balls: pd.DataFrame) -> pd.DataFrame:
    if not HOURLY.exists():
        raise FileNotFoundError(f"{HOURLY} missing: run `python -m src.features.weather` first")
    mv = balls[["match_id", "date", "venue_key"]].drop_duplicates("match_id")
    return match_weather(mv, pd.read_parquet(HOURLY), start_times())


if __name__ == "__main__":
    matches = mens_t20_matches()
    venues = build_venues(matches)
    unresolved = venues[venues.latitude.isna()]
    print(f"{len(venues)} venues, {len(unresolved)} unresolved:", unresolved.index.tolist()[:40])
    print(venues.head(60)[["city", "country", "latitude", "longitude", "source", "n_matches"]]
          .to_string())
    mv = matches.assign(venue_key=venue_key(matches.venue, matches.city))[["venue_key", "date"]]
    hourly = download_weather(venues, mv)
    print(f"hourly rows: {len(hourly):,}")
