# Evaluation report - walk-forward out-of-sample

> **DEMO RUN ON SYNTHETIC DATA.** These numbers demonstrate the report format only. Regenerate with your real trade log (`python run_pipeline.py --trades <csv>`) before submitting.

Period: 2026-07-13 to 2026-10-09 (65 business days, 6 walk-forward folds, train 90d / test 15d / step 15d, embargo 1d, top-3 selection).

## Headline metrics
| Strategy | Sharpe | Sharpe 90% CI | Sortino | Max drawdown | Total P&L | % up days |
|---|---|---|---|---|---|---|
| model | 4.12 | [1.90, 6.74] | 7.13 | -1.16 | 8.62 | 62% |
| baseline | 7.11 | [4.66, 9.33] | 16.84 | -1.13 | 15.82 | 65% |
| all_perms | 3.52 | [2.12, 6.55] | 5.74 | -0.47 | 2.53 | 63% |

![Equity curves](06_equity_curves.png)

## Prediction-quality metrics (mean across folds)
| Metric | Model | History-mean baseline |
|---|---|---|
| RMSE of trade P&L | 0.9993 | 0.9998 |
| Rank correlation (predicted vs realised permutation P&L) | 0.24 | 0.22 |
| Win-classifier AUC | 0.520 | n/a |
| Realised P&L/trade of top-3 picks | 0.135 | 0.250 |

## Definitions and caveats
- **Sharpe** = mean daily P&L / std of daily P&L x sqrt(252). Risk-free rate taken as 0. **Sortino** uses downside deviation below 0.
- **Max drawdown** is the largest peak-to-trough fall of cumulative P&L in the log's own units (additive, not a percentage of equity).
- Model and baseline each choose their permutations per fold from information available before that fold; all reported results are out-of-sample.
- No transaction costs, slippage or position sizing. Trades can overlap in time; daily P&L aggregates by entry date.
- With about 65 daily observations the Sharpe estimates are very noisy; read the bootstrap interval, not the point estimate. If intervals overlap heavily, the data do not distinguish the model from the baseline.
- Multiple permutations tested raises the chance that a good-looking result is luck; the baseline column is the honest comparison.
