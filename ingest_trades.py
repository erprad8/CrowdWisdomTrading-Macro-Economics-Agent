"""Trading-log ingestion.

Real logs: CSV with columns  trade_id, entry_time, exit_time, pnl  plus one column per strategy
parameter named  param_<name>  (e.g. param_lookback, param_stop_atr, param_tp_r). Times are parsed as UTC
unless they carry an offset. A 'permutation' = a unique combination of the param_* values.
"""
import itertools

import numpy as np
import pandas as pd

from features import build_context_features

PARAM_GRID = {"param_lookback": [10, 20, 50], "param_stop_atr": [1.0, 1.5, 2.0], "param_tp_r": [1.0, 2.0, 3.0]}


def param_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("param_")]


def load_trades(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"entry_time", "exit_time", "pnl"}
    if not need <= set(df.columns) or not param_cols(df):
        raise ValueError(f"Trade log needs {need} and at least one param_* column; got {list(df.columns)}")
    for c in ("entry_time", "exit_time"):
        df[c] = pd.to_datetime(df[c], utc=True)
    if "trade_id" not in df:
        df["trade_id"] = np.arange(len(df))
    df["win"] = (df["pnl"] > 0).astype(int)
    df["perm_id"] = df[param_cols(df)].astype(str).agg("|".join, axis=1)
    return df.sort_values("entry_time").reset_index(drop=True)


def synthetic_trades(start, end, events: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """DEMO ONLY. Simulated R-multiple P&L with a *planted* dependence on macro context, so the
    pipeline has something to find. Do not interpret demo results as market evidence."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start.date(), end.date(), tz="UTC")
    perms = list(itertools.product(*PARAM_GRID.values()))
    rows = []
    for d in days:
        for p in perms:
            for _ in range(rng.integers(0, 3)):
                entry = d + pd.Timedelta(hours=int(rng.integers(13, 20)), minutes=int(rng.integers(0, 60)))
                rows.append((entry, entry + pd.Timedelta(minutes=int(rng.integers(15, 240))), *p))
    df = pd.DataFrame(rows, columns=["entry_time", "exit_time", *PARAM_GRID])
    ctx = build_context_features(df["entry_time"], events)
    shock = (ctx["hrs_since_high_event"] <= 2).astype(float)
    mu = (0.02
          + 0.25 * (df["param_lookback"] == 50) * (ctx["sentiment_state"] < -0.3)
          + 0.25 * (df["param_tp_r"] == 3.0) * (ctx["sentiment_state"] > 0.3)
          - 0.35 * (df["param_stop_atr"] == 1.0) * shock
          + 0.15 * (df["param_stop_atr"] == 2.0) * shock)
    df["pnl"] = mu + rng.normal(0, 1.0, len(df))
    df["win"] = (df["pnl"] > 0).astype(int)
    df["trade_id"] = np.arange(len(df))
    df["perm_id"] = df[list(PARAM_GRID)].astype(str).agg("|".join, axis=1)
    return df.sort_values("entry_time").reset_index(drop=True)
