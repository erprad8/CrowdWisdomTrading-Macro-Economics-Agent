# Submission guide

## 1. Repository
    git init && git add . && git commit -m "Macro-aware strategy permutation pipeline + agents"
    git branch -M main
    git remote add origin https://github.com/<you>/crowdwisdom-macro-agent.git
    git push -u origin main
`.gitignore` already excludes `.env`, `data/`, `output/`. Commit one real-data `output/evaluation_report.md`
and its chart manually (e.g. copy into `reports/`) after step 3.

## 2. Scraping configuration (reproducibility)
| Service | Used for | Identifier |
|---|---|---|
| Apify | Macro calendar (actual/forecast/previous, UTC) | actor `scrapemint/forexfactory-economic-calendar` (default; override with `APIFY_ACTOR_ID`) |
| Tavily | Recent macro news | REST `https://api.tavily.com/search`, `topic=news` |
| Exa | Not used | - |

Do NOT put tokens in the repo. Give reviewers `.env.example`, and send real tokens through a private channel,
preferably temporary tokens you revoke after review. Also commit the first cached `data/raw/calendar_*.json`
(copy to `reports/`) so reviewers can replay the pipeline without a token (`--no-scrape`).

## 3. Evaluation report
    python run_pipeline.py --trades your_trades.csv
produces `output/evaluation_report.md` (Sharpe, Sortino, Max Drawdown with bootstrap interval, plus
prediction metrics) and `output/06_equity_curves.png`.
