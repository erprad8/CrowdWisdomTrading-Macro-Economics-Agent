"""python run_agents.py --demo     |     python run_agents.py --trades my_trades.csv"""
import argparse
import json

import config as C
from agents.team import run_team

ap = argparse.ArgumentParser()
ap.add_argument("--demo", action="store_true"); ap.add_argument("--trades")
ap.add_argument("--horizon-days", type=int, default=C.WF_TEST_DAYS)
a = ap.parse_args()
if not a.demo and not a.trades:
    ap.error("pass --demo or --trades <csv>")
state, reports = run_team(demo=a.demo, trades_path=a.trades, horizon_days=a.horizon_days)
(C.OUT / "agent_audit.json").write_text(json.dumps(state.audit, default=str, indent=1))
for k, v in reports.items():
    print(f"\n=== {k} ===\n{v}")
