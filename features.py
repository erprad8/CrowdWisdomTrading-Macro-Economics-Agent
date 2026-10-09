"""Module 2 - Feature engineering. Everything here is point-in-time: a trade at time t only sees
events released at or before t, plus the *schedule* (not outcomes) of upcoming events."""
import numpy as np
import pandas as pd

import config as C
from ingest_events import sentiment_at


def time_features(times) -> pd.DataFrame:
    ts = pd.Series(pd.to_datetime(pd.Series(times), utc=True)).reset_index(drop=True)
    est = ts.dt.tz_convert("America/New_York")
    mod = est.dt.hour * 60 + est.dt.minute
    h = ts.dt.hour
    return pd.DataFrame({
        "minute_of_day_est": mod,
        "tod_sin": np.sin(2 * np.pi * mod / 1440), "tod_cos": np.cos(2 * np.pi * mod / 1440),
        "dow": est.dt.dayofweek, "week_of_month": (est.dt.day - 1) // 7,
        "is_month_end_week": (est.dt.days_in_month - est.dt.day <= 4).astype(int),
        "session_asia": (h < 7).astype(int),
        "session_london": ((h >= 7) & (h < 13)).astype(int),
        "session_ny": ((h >= 13) & (h < 21)).astype(int),
    })


def _asof(t: pd.DataFrame, right: pd.DataFrame, cols, direction):
    """merge_asof that preserves the caller's row order."""
    right = right.sort_values("ts_utc")[["ts_utc"] + cols]
    m = pd.merge_asof(t.sort_values("t"), right, left_on="t", right_on="ts_utc",
                      direction=direction, allow_exact_matches=True)
    return m.sort_values("i").reset_index(drop=True)


def event_features(times, events: pd.DataFrame) -> pd.DataFrame:
    """Timing + surprise features. `events` must already carry sentiment_state (add_sentiment)."""
    t = pd.DataFrame({"t": pd.to_datetime(pd.Series(times), utc=True)}).reset_index(drop=True)
    t["i"] = np.arange(len(t))
    cap = C.NO_EVENT_HOURS
    released = events[events["actual"].notna() & events["family"].notna()]
    tracked_high = events[events["impact"].eq("high")]
    out = pd.DataFrame(index=range(len(t)))

    last = _asof(t, tracked_high, ["impact_w"], "backward")
    nxt = _asof(t, tracked_high, ["impact_w"], "forward")
    out["hrs_since_high_event"] = ((t["t"] - last["ts_utc"]).dt.total_seconds() / 3600).fillna(cap).clip(0, cap)
    out["hrs_to_high_event"] = ((nxt["ts_utc"] - t["t"]).dt.total_seconds() / 3600).fillna(cap).clip(0, cap)
    out["in_event_window"] = ((out["hrs_since_high_event"] <= 1) | (out["hrs_to_high_event"] <= 0.5)).astype(int)

    lr = _asof(t, released, ["surprise_z", "score"], "backward")
    out["last_surprise_z"] = lr["surprise_z"].fillna(0)
    out["last_event_score"] = lr["score"].fillna(0)

    for fam in C.CONTEXT_FAMILIES:
        f_all = events[events["family"] == fam]
        f_rel = f_all[f_all["actual"].notna()]
        b = _asof(t, f_all, [], "backward")
        f = _asof(t, f_all, [], "forward")
        out[f"hrs_since_{fam}"] = ((t["t"] - b["ts_utc"]).dt.total_seconds() / 3600).fillna(cap).clip(0, cap)
        out[f"hrs_to_{fam}"] = ((f["ts_utc"] - t["t"]).dt.total_seconds() / 3600).fillna(cap).clip(0, cap)
        s = _asof(t, f_rel, ["surprise_z"], "backward")
        out[f"last_{fam}_surprise_z"] = s["surprise_z"].fillna(0)

    out["sentiment_state"] = sentiment_at(t["t"], events).values
    out["sentiment_regime"] = np.select([out["sentiment_state"] > 0.5, out["sentiment_state"] < -0.5], [1, -1], 0)
    return out


def build_context_features(times, events: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([time_features(times), event_features(times, events)], axis=1)
