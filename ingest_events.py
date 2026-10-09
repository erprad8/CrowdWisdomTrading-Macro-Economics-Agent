"""Module 1 - Macro event ingestion: Apify calendar scrape + Tavily news."""
import json
import logging
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

import config as C

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- Apify
def fetch_calendar_apify(start: datetime, end: datetime) -> list[dict]:
    """Run the configured Apify actor for [start, end] and return raw dataset items."""
    from apify_client import ApifyClient

    if not C.APIFY_TOKEN:
        raise RuntimeError("APIFY_TOKEN is not set (see .env.example)")
    raw_input = C.APIFY_INPUT_JSON.replace("{start}", start.strftime("%Y-%m-%d")).replace(
        "{end}", end.strftime("%Y-%m-%d"))
    run_input = json.loads(raw_input)
    client = ApifyClient(C.APIFY_TOKEN)
    log.info("Running Apify actor %s with %s", C.APIFY_ACTOR_ID, run_input)
    run = client.actor(C.APIFY_ACTOR_ID).call(run_input=run_input)
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    (C.RAW / f"calendar_{stamp}.json").write_text(json.dumps(items, default=str))   # audit trail
    log.info("Apify returned %d rows", len(items))
    return items


def load_cached_calendar() -> list[dict]:
    files = sorted(C.RAW.glob("calendar_*.json"))
    if not files:
        raise FileNotFoundError("No cached calendar in data/raw; run without --no-scrape first")
    return json.loads(files[-1].read_text())


# ----------------------------------------------------------------------------- parsing
def parse_number(x) -> float:
    """'175K'->175000, '0.5%'->0.5, '5.31|2.6'->5.31, '<0.1%'->0.1, ''->nan."""
    if x is None:
        return np.nan
    s = str(x).strip().replace(",", "")
    if not s:
        return np.nan
    s = s.split("|")[0]
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return np.nan
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(s[m.end():m.end() + 1].upper(), 1.0)
    return float(m.group()) * mult


def classify_family(title: str):
    for fam, spec in C.EVENT_FAMILIES.items():
        if re.search(spec["pattern"], str(title), flags=re.I):
            return fam
    return None


def normalize_calendar(records: list[dict]) -> pd.DataFrame:
    """Map heterogeneous actor output to one schema with UTC and US/Eastern timestamps."""
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Calendar is empty")

    def pick(*cands):
        return next((c for c in cands if c in df.columns), None)

    ts_col = pick("timestamp", "dateTimeUtc", "datetimeUtc", "datetime_utc", "dateTime", "datetime")
    date_col, time_col = pick("date", "eventDate"), pick("time", "timeLabel")
    title_col = pick("eventTitle", "title", "event", "name", "eventName")
    if title_col is None or (ts_col is None and date_col is None):
        raise ValueError(f"Unrecognised calendar schema; columns={list(df.columns)}. "
                         "Add the actor's field names to normalize_calendar().")

    ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    if ts_col:
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    time_known = ts.notna()
    if date_col:
        if time_col:   # naive date+time strings are assumed to already be UTC
            combo = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str),
                                   utc=True, errors="coerce")
            time_known |= combo.notna() & ts.isna()
            ts = ts.fillna(combo)
        ts = ts.fillna(pd.to_datetime(df[date_col], utc=True, errors="coerce"))

    out = pd.DataFrame({
        "ts_utc": ts,
        "time_known": time_known.astype(int),
        "title": df[title_col].astype(str),
        "currency": df[pick("currency", "country")] if pick("currency", "country") else "USD",
        "impact": df[pick("impact", "importance")].astype(str).str.lower() if pick("impact", "importance") else "low",
        "actual": df[pick("actual")].map(parse_number) if pick("actual") else np.nan,
        "forecast": df[pick("forecast", "consensus")].map(parse_number) if pick("forecast", "consensus") else np.nan,
        "previous": df[pick("previous")].map(parse_number) if pick("previous") else np.nan,
    }).dropna(subset=["ts_utc"])

    out["impact"] = out["impact"].where(out["impact"].isin(C.IMPACT_WEIGHT), "low")
    out["impact_w"] = out["impact"].map(C.IMPACT_WEIGHT)
    out["family"] = out["title"].map(classify_family)
    out = out[(out["currency"].astype(str).str.upper() == "USD") & (out["family"].notna() | (out["impact"] == "high"))]
    out["ts_est"] = out["ts_utc"].dt.tz_convert("America/New_York")      # EST/EDT alignment
    out["surprise"] = out["actual"] - out["forecast"]
    scale = out["family"].map(lambda f: C.EVENT_FAMILIES.get(f, {}).get("scale", np.nan))
    out["surprise_z"] = (out["surprise"] / scale).clip(-5, 5)
    return out.sort_values("ts_utc").drop_duplicates(["ts_utc", "title"]).reset_index(drop=True)


def add_sentiment(events: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time macro 'risk-appetite' state: exponentially-decayed sum of signed surprises.
    Positive = risk-supportive / dovish. Negative = hawkish / risk-off."""
    ev = events.sort_values("ts_utc").reset_index(drop=True).copy()
    sign = ev["family"].map(lambda f: C.EVENT_FAMILIES.get(f, {}).get("risk_sign", 0)).astype(float)
    wt = ev["family"].map(lambda f: C.EVENT_FAMILIES.get(f, {}).get("weight", 0)).astype(float)
    ev["score"] = (sign * wt * ev["impact_w"] * ev["surprise_z"].fillna(0).clip(-3, 3)).fillna(0.0)
    hl, state, last, states = C.SENTIMENT_HALFLIFE_DAYS * 86400, 0.0, None, []
    for ts, s in zip(ev["ts_utc"], ev["score"]):
        if last is not None:
            state *= 0.5 ** ((ts - last).total_seconds() / hl)
        state += s
        last = ts
        states.append(state)
    ev["sentiment_state"] = states
    return ev


def sentiment_at(times, events: pd.DataFrame) -> pd.Series:
    """Sentiment state as of each timestamp (uses only events with ts <= t)."""
    t = pd.DataFrame({"t": pd.to_datetime(pd.Series(times), utc=True)}).reset_index().rename(columns={"index": "i"})
    ev = events[["ts_utc", "sentiment_state"]].sort_values("ts_utc")
    m = pd.merge_asof(t.sort_values("t"), ev, left_on="t", right_on="ts_utc", direction="backward")
    dt = (m["t"] - m["ts_utc"]).dt.total_seconds().fillna(0)
    val = (m["sentiment_state"].fillna(0) * 0.5 ** (dt / (C.SENTIMENT_HALFLIFE_DAYS * 86400)))
    return pd.Series(val.values, index=m["i"].values).sort_index().reset_index(drop=True)


# ----------------------------------------------------------------------------- demo calendar
def synthetic_calendar(start: datetime, end: datetime, seed: int = 7) -> list[dict]:
    """Offline stand-in shaped like a ForexFactory actor's output. DEMO ONLY - not real data."""
    rng = np.random.default_rng(seed)
    now = datetime.now(timezone.utc)
    rows = []

    def add(ts, title, impact, fc, unit="", scale=1.0, base=0.0):
        actual = fc + rng.normal(0, scale) if ts <= now else None
        fmt = (lambda v: f"{v:.1f}{unit}" if unit != "K" else f"{v:.0f}K")
        rows.append(dict(timestamp=ts.isoformat(), eventTitle=title, currency="USD", impact=impact,
                         actual=fmt(actual) if actual is not None else None, forecast=fmt(fc),
                         previous=fmt(fc - base)))

    d = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while d <= end:
        if d.weekday() == 4 and d.day <= 7:   # first Friday: jobs report
            ts = d.replace(hour=12, minute=30)
            add(ts, "Non-Farm Employment Change", "high", 180, "K", 60)
            add(ts, "Unemployment Rate", "high", 4.2, "%", 0.1)
            add(ts, "Average Hourly Earnings m/m", "medium", 0.3, "%", 0.1)
        if d.day == 12 and d.weekday() < 5:
            ts = d.replace(hour=12, minute=30)
            add(ts, "CPI m/m", "high", 0.3, "%", 0.1)
            add(ts, "Core CPI m/m", "high", 0.3, "%", 0.1)
        d += timedelta(days=1)
    d = start.replace(hour=18, minute=0, second=0, microsecond=0)
    while d <= end:                           # ~6-weekly FOMC (Wednesdays)
        d += timedelta(days=(2 - d.weekday()) % 7)
        add(d, "Federal Funds Rate", "high", 4.5, "%", 0.12)
        rows.append(dict(timestamp=d.isoformat(), eventTitle="FOMC Statement", currency="USD",
                         impact="high", actual=None, forecast=None, previous=None))
        d += timedelta(days=42)
    return rows


# ----------------------------------------------------------------------------- Tavily news
HAWKISH = ["rate hike", "hawkish", "hotter", "sticky inflation", "higher for longer", "tightening",
           "stronger than expected", "accelerat", "beat expectations", "inflation surge", "overheat"]
DOVISH = ["rate cut", "dovish", "cooling", "disinflation", "softer", "slowdown", "easing",
          "weaker than expected", "missed expectations", "layoffs", "pause"]


def lexicon_sentiment(text: str) -> float:
    """Crude lexicon score in [-1, 1]; +1 = dovish/risk-supportive. A placeholder for a real NLP model."""
    t = text.lower()
    d, h = sum(k in t for k in DOVISH), sum(k in t for k in HAWKISH)
    return 0.0 if d + h == 0 else (d - h) / (d + h)


def fetch_news_tavily(queries=None, days: int = 7, max_results: int = 8) -> pd.DataFrame:
    if not C.TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set (see .env.example)")
    queries = queries or ["US CPI inflation Fed outlook", "FOMC rate decision expectations",
                          "US jobs report nonfarm payrolls labor market", "US economy macro outlook markets"]
    rows = []
    for q in queries:
        r = requests.post("https://api.tavily.com/search",
                          headers={"Authorization": f"Bearer {C.TAVILY_API_KEY}"},
                          json={"query": q, "topic": "news", "days": days, "max_results": max_results,
                                "search_depth": "basic"}, timeout=60)
        r.raise_for_status()
        for x in r.json().get("results", []):
            rows.append(dict(query=q, title=x.get("title", ""), url=x.get("url", ""),
                             content=x.get("content", ""), published=x.get("published_date"),
                             tavily_score=x.get("score")))
    df = pd.DataFrame(rows).drop_duplicates("url")
    if not df.empty:
        df["sentiment"] = (df["title"] + ". " + df["content"]).map(lexicon_sentiment)
    return df


def synthetic_news() -> pd.DataFrame:
    """Placeholder headlines for --demo only."""
    items = [("DEMO: Inflation cooling but services prices sticky, Fed signals patience", 0.0),
             ("DEMO: Payrolls come in softer than expected, rate cut bets rise", 1.0),
             ("DEMO: Hotter CPI revives higher for longer concerns", -1.0)]
    return pd.DataFrame([dict(query="demo", title=t, url="", content=t, published=None,
                              tavily_score=None, sentiment=s) for t, s in items])
