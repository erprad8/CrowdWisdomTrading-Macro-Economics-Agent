"""SQLite store: events, trades, news, and the joined point-in-time model table."""
import sqlite3

import pandas as pd

import config as C

FMT = "%Y-%m-%d %H:%M:%S"
TIME_COLS = ["ts_utc", "ts_est", "entry_time", "exit_time", "published"]


def connect():
    return sqlite3.connect(C.DB_PATH)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if str(df[c].dtype).startswith("datetime64"):
            df[c] = (df[c].dt.tz_convert("UTC") if df[c].dt.tz is not None else df[c]).dt.strftime(FMT)
    return df


def save(name: str, df: pd.DataFrame):
    with connect() as con:
        _prep(df).to_sql(name, con, if_exists="replace", index=False)


def load(name: str) -> pd.DataFrame:
    with connect() as con:
        df = pd.read_sql(f"SELECT * FROM {name}", con)
    for c in TIME_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df


VIEW_SQL = """
DROP VIEW IF EXISTS v_trade_last_event;
CREATE VIEW v_trade_last_event AS
SELECT t.trade_id, t.entry_time, e.title AS last_event, e.surprise_z AS last_surprise_z
FROM trades t
LEFT JOIN events e ON e.rowid = (
    SELECT e2.rowid FROM events e2
    WHERE e2.ts_utc <= t.entry_time AND e2.actual IS NOT NULL AND e2.family IS NOT NULL
    ORDER BY e2.ts_utc DESC LIMIT 1);
"""


def create_views():
    """Pure-SQL point-in-time join (useful for ad-hoc inspection and as a cross-check of features.py)."""
    with connect() as con:
        con.executescript(VIEW_SQL)
