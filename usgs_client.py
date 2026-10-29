"""
usgs_client.py
Data-access layer for the USGS Earthquake Catalog API (fdsnws) — free,
public, no API key required.

Docs: https://earthquake.usgs.gov/fdsnws/event/1/
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
SIGNIFICANT_MONTH_FEED = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.geojson"
SIGNIFICANT_WEEK_FEED = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_week.geojson"


class USGSClientError(RuntimeError):
    pass


def _features_to_df(features: list) -> pd.DataFrame:
    rows = []
    for f in features:
        props = f.get("properties", {})
        coords = f.get("geometry", {}).get("coordinates", [None, None, None])
        rows.append({
            "id": f.get("id"),
            "time": pd.to_datetime(props.get("time"), unit="ms", utc=True),
            "magnitude": props.get("mag"),
            "place": props.get("place"),
            "longitude": coords[0],
            "latitude": coords[1],
            "depth_km": coords[2],
            "url": props.get("url"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("time").reset_index(drop=True)
    return df


def fetch_significant_earthquakes(period: str = "month") -> pd.DataFrame:
    """Fetch USGS's curated list of significant recent earthquakes — good
    candidates to pick as a 'mainshock' to forecast around."""
    url = SIGNIFICANT_MONTH_FEED if period == "month" else SIGNIFICANT_WEEK_FEED
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise USGSClientError(f"Could not reach the USGS earthquake feed: {e}") from e
    data = resp.json()
    return _features_to_df(data.get("features", []))


def fetch_catalog(
    starttime: datetime,
    endtime: datetime,
    min_magnitude: float = 2.5,
    latitude: float | None = None,
    longitude: float | None = None,
    max_radius_km: float | None = None,
    limit: int = 20000,
) -> pd.DataFrame:
    """Query the full earthquake catalog for a time window, optionally
    restricted to a circular region (for pulling a mainshock's aftershock
    sequence)."""
    params = {
        "format": "geojson",
        "starttime": starttime.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": endtime.strftime("%Y-%m-%dT%H:%M:%S"),
        "minmagnitude": min_magnitude,
        "orderby": "time-asc",
        "limit": limit,
    }
    if latitude is not None and longitude is not None and max_radius_km is not None:
        params.update({"latitude": latitude, "longitude": longitude, "maxradiuskm": max_radius_km})

    try:
        resp = requests.get(QUERY_URL, params=params, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise USGSClientError(f"Could not fetch the earthquake catalog: {e}") from e
    data = resp.json()
    return _features_to_df(data.get("features", []))


def fetch_aftershock_sequence(
    mainshock_time: datetime,
    mainshock_lat: float,
    mainshock_lon: float,
    radius_km: float = 200,
    days_after: int = 30,
    min_magnitude: float = 2.5,
) -> pd.DataFrame:
    """Fetch events in a radius/time window after a given mainshock,
    excluding the mainshock's own timestamp."""
    end = mainshock_time + timedelta(days=days_after)
    df = fetch_catalog(mainshock_time, end, min_magnitude=min_magnitude,
                        latitude=mainshock_lat, longitude=mainshock_lon, max_radius_km=radius_km)
    if df.empty:
        return df
    # drop anything at/before the mainshock instant (the mainshock itself, if it matched the query).
    # mainshock_time may arrive naive or already tz-aware depending on the caller, so normalize
    # explicitly rather than passing straight into pd.Timestamp(..., tz=...), which raises if the
    # input already carries tzinfo.
    ms_ts = pd.Timestamp(mainshock_time)
    ms_ts = ms_ts.tz_localize("UTC") if ms_ts.tzinfo is None else ms_ts.tz_convert("UTC")
    df = df[df["time"] > ms_ts + pd.Timedelta(seconds=1)]
    return df.reset_index(drop=True)
