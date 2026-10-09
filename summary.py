"""One-page macro outlook (Markdown), built from the data - no free-text generation."""
import numpy as np
import pandas as pd

import config as C
from ingest_events import sentiment_at


def regime_label(v):
    return "RISK-SUPPORTIVE / DOVISH" if v > .5 else "HAWKISH / RISK-OFF" if v < -.5 else "NEUTRAL"


def build_summary(events, news, rec, folds, demo: bool) -> str:
    now = pd.Timestamp.now(tz="UTC")
    cur = float(sentiment_at([now], events).iloc[0])
    ago = float(sentiment_at([now - pd.Timedelta(days=30)], events).iloc[0])
    rel = events[events["actual"].notna() & events["family"].notna() & events["surprise_z"].notna()].tail(8)
    up = events[(events["ts_utc"] > now) & (events["ts_utc"] <= now + pd.Timedelta(days=14)) & (events["impact"] == "high")].head(8)

    L = [f"# Macro Outlook - {now:%d %b %Y}", ""]
    if demo:
        L += ["> **DEMO MODE:** events, news and trades are synthetic. Do not trade on this page.", ""]
    L += [f"**Regime:** {regime_label(cur)} (state {cur:+.2f}; 30 days ago {ago:+.2f}, change {cur - ago:+.2f}). "
          "State = decayed sum of signed CPI/NFP/FOMC/labour surprises; sign convention assumes strong data = tighter policy.", ""]
    L += ["## Latest releases (actual vs forecast)", "| Date (UTC) | Event | Actual | Forecast | Surprise (z) |", "|---|---|---|---|---|"]
    for _, r in rel.iterrows():
        L.append(f"| {r.ts_utc:%d %b} | {r.title} | {r.actual:g} | {r.forecast:g} | {r.surprise_z:+.1f} |")
    L += ["", "## Next 14 days - high-impact (UTC / US Eastern)"]
    L += [f"- {r.ts_utc:%a %d %b %H:%M} / {r.ts_est:%H:%M} - {r.title}" + (f" (fcst {r.forecast:g})" if pd.notna(r.forecast) else "")
          for _, r in up.iterrows()] or ["- none scheduled in the scraped window"]
    if news is not None and len(news):
        n = news.sort_values("tavily_score", ascending=False, na_position="last").head(5)
        L += ["", f"## News flow (Tavily; mean lexicon tone {news['sentiment'].mean():+.2f}, +ve = dovish)"]
        L += [f"- {r.title} ({r.sentiment:+.0f})" for _, r in n.iterrows()]
    top = rec.head(3)
    L += ["", f"## Model view - top permutations, next horizon (walk-forward OOS over {len(folds)} windows)"]
    L += [f"- `{r.perm_id}`: predicted P&L/trade {r.pred_pnl:+.3f}, win prob {r.pred_win:.1%}" for _, r in top.iterrows()]
    L += [f"- OOS evidence: model top-{C.TOP_K} averaged {folds['topk_model'].mean():+.3f}/trade vs {folds['topk_base'].mean():+.3f} for "
          f"history-mean and {folds['all_perm_avg'].mean():+.3f} for all permutations; rank-corr {folds['rank_corr_model'].mean():+.2f} vs {folds['rank_corr_base'].mean():+.2f}.",
          "", "*Research output, not investment advice. Sentiment weights and the lexicon news score are simple heuristics.*"]
    return "\n".join(L)
