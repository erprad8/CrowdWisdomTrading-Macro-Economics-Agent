"""Module 4 - Visual matrix / chart generator."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore', message='.*tight_layout.*')

import config as C
from ingest_events import sentiment_at

FAM_COLOR = {"CPI": "#d62728", "NFP": "#1f77b4", "FOMC": "#2ca02c", "UNEMPLOYMENT": "#9467bd", "WAGES": "#8c564b"}


def _save(fig, name):
    p = C.OUT / name
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def sentiment_timeline(events: pd.DataFrame):
    """Macro sentiment state through time, with event markers (size = |surprise|)."""
    now = pd.Timestamp.now(tz="UTC")
    days = pd.date_range(events["ts_utc"].min().normalize(), min(events["ts_utc"].max(), now), freq="6h", tz="UTC")
    s = sentiment_at(days, events)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(days, s, color="black", lw=1.4)
    ax.fill_between(days, s, 0, where=s >= 0, color="#2ca02c", alpha=.25, label="Risk-supportive / dovish")
    ax.fill_between(days, s, 0, where=s < 0, color="#d62728", alpha=.25, label="Hawkish / risk-off")
    rel = events[events["actual"].notna() & events["family"].notna()]
    for fam, g in rel.groupby("family"):
        y = sentiment_at(g["ts_utc"], events).values
        ax.scatter(g["ts_utc"], y, s=20 + 40 * g["surprise_z"].abs().fillna(0), color=FAM_COLOR.get(fam, "grey"),
                   edgecolor="white", zorder=3, label=fam)
    ax.axhline(0, color="grey", lw=.8)
    ax.set_title("Macro sentiment state over time (decayed sum of signed surprises)")
    ax.set_ylabel("Sentiment state"); ax.legend(ncol=4, fontsize=8, loc="upper left")
    return _save(fig, "01_macro_sentiment_timeline.png")


def surprise_chart(events: pd.DataFrame):
    rel = events[events["actual"].notna() & events["family"].isin(["CPI", "NFP", "FOMC", "UNEMPLOYMENT"])]
    fams = [f for f in ["CPI", "NFP", "UNEMPLOYMENT", "FOMC"] if f in set(rel["family"])]
    fig, axes = plt.subplots(len(fams), 1, figsize=(12, 2.2 * len(fams)), sharex=True, squeeze=False)
    for ax, fam in zip(axes[:, 0], fams):
        g = rel[rel["family"] == fam]
        ax.bar(g["ts_utc"], g["surprise_z"], width=2, color=[FAM_COLOR[fam] if v >= 0 else "#999" for v in g["surprise_z"]])
        ax.axhline(0, color="black", lw=.6); ax.set_ylabel(fam, fontsize=9)
    axes[0, 0].set_title("Event surprises (actual - forecast, in units of typical surprise)")
    return _save(fig, "02_event_surprises.png")


def permutation_matrix(preds: pd.DataFrame):
    """Rows = strategy permutations, columns = walk-forward test windows. Predicted vs realised mean P&L."""
    g = preds.groupby(["perm_id", "fold"]).agg(real=("pnl", "mean"), pred=("pred_pnl", "mean")).reset_index()
    order = g.groupby("perm_id")["real"].mean().sort_values(ascending=False).index
    R = g.pivot(index="perm_id", columns="fold", values="real").loc[order]
    P = g.pivot(index="perm_id", columns="fold", values="pred").loc[order]
    v = float(np.nanpercentile(np.abs(np.r_[R.values.ravel(), P.values.ravel()]), 95))
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, .28 * len(R))), sharey=True)
    for ax, M, t in zip(axes, (P, R), ("Model-predicted mean P&L", "Realised mean P&L")):
        im = ax.imshow(M.values, aspect="auto", cmap="RdYlGn", vmin=-v, vmax=v)
        ax.set_title(t); ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns)
        ax.set_xlabel("Walk-forward window (out-of-sample)")
    axes[0].set_yticks(range(len(R))); axes[0].set_yticklabels(R.index, fontsize=7)
    axes[0].set_ylabel("Permutation (lookback|stop_atr|tp_r)")
    fig.colorbar(im, ax=axes, shrink=.6, label="P&L per trade")
    return _save(fig, "03_permutation_matrix_walkforward.png")


def regime_matrix(df: pd.DataFrame):
    """Mean realised P&L per permutation under each macro-sentiment regime (the 'sentiment change' matrix)."""
    d = df.assign(regime=df["sentiment_regime"].map({-1: "Hawkish / risk-off", 0: "Neutral", 1: "Risk-supportive"}))
    M = d.pivot_table(index="perm_id", columns="regime", values="pnl", aggfunc="mean")
    M = M.reindex(columns=[c for c in ["Hawkish / risk-off", "Neutral", "Risk-supportive"] if c in M.columns])
    M = M.loc[M.mean(axis=1).sort_values(ascending=False).index]
    v = float(np.nanpercentile(np.abs(M.values), 95))
    fig, ax = plt.subplots(figsize=(7, max(6, .28 * len(M))))
    im = ax.imshow(M.values, aspect="auto", cmap="RdYlGn", vmin=-v, vmax=v)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, rotation=20)
    ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index, fontsize=7)
    ax.set_title("Mean P&L by permutation and macro-sentiment regime")
    fig.colorbar(im, ax=ax, label="P&L per trade")
    return _save(fig, "04_regime_matrix.png")


def walkforward_performance(folds: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = folds["fold"]
    axes[0].plot(x, folds["topk_model"], "o-", label=f"Model top-{C.TOP_K}")
    axes[0].plot(x, folds["topk_base"], "s--", label=f"History-mean top-{C.TOP_K}")
    axes[0].plot(x, folds["all_perm_avg"], color="grey", label="All-permutation average")
    axes[0].axhline(0, color="black", lw=.6)
    axes[0].set_title("Realised P&L of chosen permutations, per OOS window"); axes[0].set_xlabel("Fold"); axes[0].legend(fontsize=8)
    axes[1].bar(x - .2, folds["rank_corr_model"], .4, label="Model")
    axes[1].bar(x + .2, folds["rank_corr_base"], .4, label="History mean")
    axes[1].axhline(0, color="black", lw=.6)
    axes[1].set_title("Rank correlation: predicted vs realised permutation P&L"); axes[1].set_xlabel("Fold"); axes[1].legend(fontsize=8)
    return _save(fig, "05_walkforward_performance.png")


def make_all(events, df, preds, folds):
    return [sentiment_timeline(events), surprise_chart(events), permutation_matrix(preds),
            regime_matrix(df), walkforward_performance(folds)]
