"""Module 3 - Predictive model + walk-forward validation + forward recommendation."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

import config as C
from features import build_context_features
from ingest_trades import param_cols

ID_COLS = {"trade_id", "entry_time", "exit_time", "pnl", "win", "perm_id"}


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS and pd.api.types.is_numeric_dtype(df[c])]


def _fit(train: pd.DataFrame, feats):
    reg = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=60,
                                        min_samples_leaf=150, l2_regularization=5.0, random_state=0)
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=60,
                                         min_samples_leaf=150, l2_regularization=5.0, random_state=0)
    reg.fit(train[feats], train["pnl"])
    clf.fit(train[feats], train["win"])
    return reg, clf


def _rank_corr(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank())) if len(a) > 2 else np.nan


def walk_forward(df: pd.DataFrame, min_perm_trades: int = 5):
    """Rolling-window walk-forward with purge + embargo.
    Train: trades whose EXIT is before (test_start - embargo) and entry within WF_TRAIN_DAYS.
    Test : trades entering in [test_start, test_start + WF_TEST_DAYS). Nothing from the test window leaks."""
    feats = feature_cols(df)
    t0, t1 = df["entry_time"].min(), df["entry_time"].max()
    test_start = t0 + pd.Timedelta(days=C.WF_TRAIN_DAYS)
    preds, folds = [], []
    k = 0
    while test_start < t1:
        test_end = test_start + pd.Timedelta(days=C.WF_TEST_DAYS)
        cutoff = test_start - pd.Timedelta(days=C.WF_EMBARGO_DAYS)
        train = df[(df["exit_time"] < cutoff) & (df["entry_time"] >= cutoff - pd.Timedelta(days=C.WF_TRAIN_DAYS))]
        test = df[(df["entry_time"] >= test_start) & (df["entry_time"] < test_end)].copy()
        if len(train) > 300 and len(test) > 50:
            reg, clf = _fit(train, feats)
            test["pred_pnl"] = reg.predict(test[feats])
            test["pred_win"] = clf.predict_proba(test[feats])[:, 1]
            base = train.groupby("perm_id")["pnl"].mean()                       # naive benchmark
            test["base_pnl"] = test["perm_id"].map(base).fillna(train["pnl"].mean())
            test["fold"] = k
            g = test.groupby("perm_id").agg(real=("pnl", "mean"), pred=("pred_pnl", "mean"),
                                            base=("base_pnl", "mean"), n=("pnl", "size"))
            g = g[g["n"] >= min_perm_trades]
            top = lambda col: g.sort_values(col, ascending=False).head(C.TOP_K)["real"].mean()
            auc = roc_auc_score(test["win"], test["pred_win"]) if test["win"].nunique() > 1 else np.nan
            folds.append(dict(
                fold=k, test_start=test_start, n_train=len(train), n_test=len(test),
                rmse_model=mean_squared_error(test["pnl"], test["pred_pnl"]) ** 0.5,
                rmse_base=mean_squared_error(test["pnl"], test["base_pnl"]) ** 0.5,
                mae_model=mean_absolute_error(test["pnl"], test["pred_pnl"]),
                mae_base=mean_absolute_error(test["pnl"], test["base_pnl"]),
                auc_win=auc, rank_corr_model=_rank_corr(g["real"], g["pred"]),
                rank_corr_base=_rank_corr(g["real"], g["base"]),
                topk_model=top("pred"), topk_base=top("base"), all_perm_avg=g["real"].mean()))
            preds.append(test)
            k += 1
        test_start += pd.Timedelta(days=C.WF_STEP_DAYS)
    if not folds:
        raise RuntimeError("No valid folds: need more history or smaller WF_* windows in config.py")
    return pd.concat(preds), pd.DataFrame(folds)


def final_model(df: pd.DataFrame):
    feats = feature_cols(df)
    cutoff = df["entry_time"].max()
    train = df[(df["exit_time"] <= cutoff) & (df["entry_time"] >= cutoff - pd.Timedelta(days=C.WF_TRAIN_DAYS))]
    reg, clf = _fit(train, feats)
    return reg, clf, feats


def recommend(models, events: pd.DataFrame, perms: pd.DataFrame, horizon_days: int, hours=(14, 15, 16, 17, 18, 19)):
    """Score every permutation across the next `horizon_days` of business days using the *scheduled* calendar."""
    reg, clf, feats = models
    start = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1)
    days = pd.bdate_range(start, start + pd.Timedelta(days=horizon_days), tz="UTC")
    times = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in hours])
    ctx = build_context_features(times, events)
    ctx["day"] = times.normalize()
    day_has_event = events[events["impact"].eq("high")]["ts_utc"].dt.normalize().unique()
    ctx["event_day"] = ctx["day"].isin(day_has_event).astype(int)
    rows = []
    for _, p in perms.iterrows():
        x = ctx.copy()
        for c in perms.columns:
            x[c] = p[c]
        x = x[feats]
        rows.append(dict(**p.to_dict(), pred_pnl=reg.predict(x).mean(), pred_win=clf.predict_proba(x)[:, 1].mean(),
                         pred_pnl_event_days=reg.predict(x[ctx["event_day"] == 1]).mean() if (ctx["event_day"] == 1).any() else np.nan,
                         pred_pnl_quiet_days=reg.predict(x[ctx["event_day"] == 0]).mean() if (ctx["event_day"] == 0).any() else np.nan))
    return pd.DataFrame(rows).sort_values("pred_pnl", ascending=False).reset_index(drop=True), ctx
