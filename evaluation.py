"""Out-of-sample trading evaluation: Sharpe, Sortino, Max Drawdown for model-selected permutations vs benchmarks.

Method (per walk-forward fold, using only that fold's out-of-sample trades):
  model     = the TOP_K permutations ranked by the model's mean predicted P&L
  baseline  = the TOP_K permutations ranked by their historical mean P&L in the training window
  all_perms = every permutation (equal weight)
Capital is split equally across the selected permutations, so daily P&L = (sum of selected trades' P&L that day)
/ (number of selected permutations). Days without trades count as 0. Units are those of the trade log's `pnl`
(R-multiples in demo mode). No costs, slippage or position sizing are modelled.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

ANN = 252


def _select(preds: pd.DataFrame, min_n: int = 5) -> dict:
    parts = {"model": [], "baseline": [], "all_perms": []}
    for _, g in preds.groupby("fold"):
        s = g.groupby("perm_id").agg(pred=("pred_pnl", "mean"), base=("base_pnl", "mean"), n=("pnl", "size"))
        s = s[s["n"] >= min_n]
        for name, ids in (("model", s.sort_values("pred", ascending=False).head(C.TOP_K).index),
                          ("baseline", s.sort_values("base", ascending=False).head(C.TOP_K).index),
                          ("all_perms", s.index)):
            sel = g[g["perm_id"].isin(ids)].copy()
            sel["w"] = 1.0 / max(len(ids), 1)
            parts[name].append(sel)
    return {k: pd.concat(v) for k, v in parts.items()}


def daily_pnl(sel: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    d = (sel["pnl"] * sel["w"]).groupby(sel["entry_time"].dt.tz_convert("UTC").dt.normalize()).sum()
    return d.reindex(index, fill_value=0.0)


def metrics(daily: pd.Series) -> dict:
    mu, sd = daily.mean(), daily.std(ddof=1)
    downside = float(np.sqrt((np.minimum(daily, 0) ** 2).mean()))
    eq = daily.cumsum()
    peak = np.maximum.accumulate(np.r_[0.0, eq.values])[1:]
    return dict(sharpe=mu / sd * np.sqrt(ANN) if sd > 0 else np.nan,
                sortino=mu / downside * np.sqrt(ANN) if downside > 0 else np.nan,
                max_drawdown=float((eq.values - peak).min()), total_pnl=float(eq.iloc[-1]),
                avg_daily=float(mu), pct_up_days=float((daily > 0).mean()), n_days=int(len(daily)))


def sharpe_ci(daily: pd.Series, block: int = 5, n: int = 2000, seed: int = 0):
    """Moving-block bootstrap 90% interval for the annualised Sharpe."""
    rng, x = np.random.default_rng(seed), daily.values
    nb = int(np.ceil(len(x) / block))
    out = []
    for _ in range(n):
        starts = rng.integers(0, len(x) - block + 1, nb)
        s = np.concatenate([x[i:i + block] for i in starts])[:len(x)]
        out.append(s.mean() / s.std(ddof=1) * np.sqrt(ANN) if s.std(ddof=1) > 0 else np.nan)
    return tuple(np.nanpercentile(out, [5, 95]))


def write_report(preds: pd.DataFrame, folds: pd.DataFrame, demo: bool):
    sel = _select(preds)
    lo, hi = preds["entry_time"].min().normalize(), preds["entry_time"].max().normalize()
    idx = pd.bdate_range(lo.tz_convert(None), hi.tz_convert(None), tz="UTC")
    series = {k: daily_pnl(v, idx) for k, v in sel.items()}
    rows = {k: metrics(v) for k, v in series.items()}
    ci = {k: sharpe_ci(v) for k, v in series.items()}

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for k, v in series.items():
        eq = v.cumsum()
        ax[0].plot(eq.index, eq, label=k)
        pk = np.maximum.accumulate(np.r_[0.0, eq.values])[1:]
        ax[1].plot(eq.index, eq.values - pk, label=k)
    ax[0].set_title("Out-of-sample cumulative P&L (walk-forward, equal capital per selected permutation)")
    ax[0].legend(); ax[1].set_title("Drawdown"); ax[1].set_ylabel("P&L units")
    fig.tight_layout(); fig.savefig(C.OUT / "06_equity_curves.png", dpi=150); plt.close(fig)

    f = lambda x, p=2: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}"
    L = ["# Evaluation report - walk-forward out-of-sample", ""]
    if demo:
        L += ["> **DEMO RUN ON SYNTHETIC DATA.** These numbers demonstrate the report format only. Regenerate with your real "
              "trade log (`python run_pipeline.py --trades <csv>`) before submitting.", ""]
    L += [f"Period: {lo:%Y-%m-%d} to {hi:%Y-%m-%d} ({len(idx)} business days, {len(folds)} walk-forward folds, "
          f"train {C.WF_TRAIN_DAYS}d / test {C.WF_TEST_DAYS}d / step {C.WF_STEP_DAYS}d, embargo {C.WF_EMBARGO_DAYS}d, top-{C.TOP_K} selection).", "",
          "## Headline metrics", "| Strategy | Sharpe | Sharpe 90% CI | Sortino | Max drawdown | Total P&L | % up days |", "|---|---|---|---|---|---|---|"]
    for k, m in rows.items():
        L.append(f"| {k} | {f(m['sharpe'])} | [{f(ci[k][0])}, {f(ci[k][1])}] | {f(m['sortino'])} | {f(m['max_drawdown'])} | "
                 f"{f(m['total_pnl'])} | {m['pct_up_days']:.0%} |")
    L += ["", "![Equity curves](06_equity_curves.png)", "",
          "## Prediction-quality metrics (mean across folds)", "| Metric | Model | History-mean baseline |", "|---|---|---|",
          f"| RMSE of trade P&L | {f(folds.rmse_model.mean(), 4)} | {f(folds.rmse_base.mean(), 4)} |",
          f"| Rank correlation (predicted vs realised permutation P&L) | {f(folds.rank_corr_model.mean())} | {f(folds.rank_corr_base.mean())} |",
          f"| Win-classifier AUC | {f(folds.auc_win.mean(), 3)} | n/a |",
          f"| Realised P&L/trade of top-{C.TOP_K} picks | {f(folds.topk_model.mean(), 3)} | {f(folds.topk_base.mean(), 3)} |", "",
          "## Definitions and caveats",
          f"- **Sharpe** = mean daily P&L / std of daily P&L x sqrt({ANN}). Risk-free rate taken as 0. **Sortino** uses downside deviation below 0.",
          "- **Max drawdown** is the largest peak-to-trough fall of cumulative P&L in the log's own units (additive, not a percentage of equity).",
          "- Model and baseline each choose their permutations per fold from information available before that fold; all reported results are out-of-sample.",
          "- No transaction costs, slippage or position sizing. Trades can overlap in time; daily P&L aggregates by entry date.",
          f"- With about {len(idx)} daily observations the Sharpe estimates are very noisy; read the bootstrap interval, not the point estimate. "
          "If intervals overlap heavily, the data do not distinguish the model from the baseline.",
          "- Multiple permutations tested raises the chance that a good-looking result is luck; the baseline column is the honest comparison.", ""]
    (C.OUT / "evaluation_report.md").write_text("\n".join(L))
    return rows
