# Macro-aware strategy-permutation predictor

Pipeline: Apify calendar + Tavily news + trade logs -> SQLite -> point-in-time features -> gradient-boosted
P&L / win-rate model -> walk-forward validation -> ranked permutations + 1-page outlook + chart matrix.

## Run
    pip install -r requirements.txt
    python run_pipeline.py --demo                      # offline, synthetic data, no keys
    cp .env.example .env                               # add APIFY_TOKEN, TAVILY_API_KEY
    python run_pipeline.py --trades my_trades.csv      # real run (add --no-scrape to reuse cached Apify data)

## Trade log format
CSV: `trade_id, entry_time, exit_time, pnl, param_<name>...` (one `param_*` column per strategy parameter).
A permutation is a unique combination of the `param_*` values.

## Before the first real run (not verifiable offline)
1. Check the chosen Apify actor's **input schema** and edit `APIFY_INPUT_JSON` keys in `.env`.
2. Inspect the first cached `data/raw/calendar_*.json`; if field names differ, extend `normalize_calendar()`.
3. Confirm the actor's timestamps are UTC (the code assumes naive times are UTC).
4. Confirm Tavily's `days` parameter behaviour for your plan.

## Method notes
- **No look-ahead:** features use only events released at or before entry time, plus the *schedule* of upcoming events.
  Surprise z-scores use fixed per-family scales, not full-sample statistics.
- **Walk-forward:** rolling 90-day train, 15-day test, step 15; trades are purged if they exit after the cutoff; 1-day embargo.
- **Benchmark:** every fold compares the model to "each permutation's historical mean P&L". Beating this is the real test.
- **Sentiment:** decayed sum of signed surprises. The sign convention (`risk_sign` in `config.py`) is an assumption; flip it if
  your regime is "good news is good news". News tone is a crude lexicon score and is NOT a model feature (no history).
- Only 180 days of data means few CPI/FOMC releases; expect noisy fold-to-fold results. Extend `LOOKBACK_DAYS` if you can.

## Outputs (`output/`)
`macro_outlook.md`, `recommended_permutations.csv`, `walkforward_folds.csv`, `walkforward_metrics.json`,
`oos_predictions.csv`, and charts 01-05 (sentiment timeline, surprises, permutation x window matrix, regime matrix, walk-forward performance).
