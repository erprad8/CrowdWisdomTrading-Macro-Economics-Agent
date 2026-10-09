"""End-to-end runner.   python run_pipeline.py --demo     (offline, synthetic data)
                       python run_pipeline.py --trades my_trades.csv   (real: needs APIFY_TOKEN, TAVILY_API_KEY)"""
import argparse
import json
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

import config as C
import database as db
import evaluation as E
import ingest_events as ie
import ingest_trades as it
import model as M
import summary as S
import visuals as V
from features import build_context_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic calendar/news/trades, no API keys needed")
    ap.add_argument("--trades", help="CSV trading log (see ingest_trades.py)")
    ap.add_argument("--no-scrape", action="store_true", help="reuse the latest cached Apify run in data/raw")
    ap.add_argument("--horizon-days", type=int, default=C.WF_TEST_DAYS)
    a = ap.parse_args()
    if not a.demo and not a.trades:
        ap.error("pass --demo or --trades <csv>")

    now = datetime.now(timezone.utc)
    start, end = now - timedelta(days=C.LOOKBACK_DAYS), now + timedelta(days=a.horizon_days + 7)

    # 1. Macro events + news -------------------------------------------------------------
    raw = ie.synthetic_calendar(start - timedelta(days=45), end) if a.demo else (
        ie.load_cached_calendar() if a.no_scrape else ie.fetch_calendar_apify(start, end))
    events = ie.add_sentiment(ie.normalize_calendar(raw))
    news = ie.synthetic_news() if a.demo else ie.fetch_news_tavily()
    log.info("events=%d (released=%d)  news=%d", len(events), events["actual"].notna().sum(), len(news))

    # 2. Trades --------------------------------------------------------------------------
    trades = it.synthetic_trades(start, now, events) if a.demo else it.load_trades(a.trades)
    trades = trades[trades["entry_time"] >= pd.Timestamp(start)].reset_index(drop=True)

    # 3. Database join + features --------------------------------------------------------
    db.save("events", events); db.save("trades", trades); db.save("news", news)
    db.create_views()
    ctx = build_context_features(trades["entry_time"], events)
    model_table = pd.concat([trades.reset_index(drop=True), ctx], axis=1)
    db.save("model_table", model_table)
    df = db.load("model_table")
    log.info("model_table rows=%d, permutations=%d", len(df), df["perm_id"].nunique())

    # 4. Walk-forward validation ---------------------------------------------------------
    preds, folds = M.walk_forward(df)
    folds.to_csv(C.OUT / "walkforward_folds.csv", index=False)
    preds[["trade_id", "entry_time", "perm_id", "fold", "pnl", "pred_pnl", "pred_win", "base_pnl"]].to_csv(C.OUT / "oos_predictions.csv", index=False)
    agg = folds[["rmse_model", "rmse_base", "auc_win", "rank_corr_model", "rank_corr_base", "topk_model", "topk_base", "all_perm_avg"]].mean().round(4).to_dict()
    (C.OUT / "walkforward_metrics.json").write_text(json.dumps(agg, indent=2))
    log.info("OOS means: %s", agg)

    E.write_report(preds, folds, a.demo)

    # 5. Forward recommendation ----------------------------------------------------------
    pcols = it.param_cols(df)
    perms = df[pcols + ["perm_id"]].drop_duplicates().reset_index(drop=True)
    rec, _ = M.recommend(M.final_model(df), events, perms[pcols], a.horizon_days)
    rec["perm_id"] = rec[pcols].astype(str).agg("|".join, axis=1)
    rec.to_csv(C.OUT / "recommended_permutations.csv", index=False)

    # 6. Summary + charts ----------------------------------------------------------------
    (C.OUT / "macro_outlook.md").write_text(S.build_summary(events, news, rec, folds, a.demo))
    charts = V.make_all(events, df, preds, folds)
    log.info("Wrote %s", [p.name for p in charts])
    print("\nTop permutations for the next horizon:\n", rec.head(5).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
