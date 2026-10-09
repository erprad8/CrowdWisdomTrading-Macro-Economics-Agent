"""Central configuration: paths, API settings, and the macro-event taxonomy."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
OUT = ROOT / "output"
for _p in (RAW, OUT):
    _p.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA / "macro.db"

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", 180))

APIFY_TOKEN = os.getenv("APIFY_TOKEN", "")
APIFY_ACTOR_ID = os.getenv("APIFY_ACTOR_ID", "scrapemint/forexfactory-economic-calendar")
APIFY_INPUT_JSON = os.getenv("APIFY_INPUT_JSON", '{"startDate": "{start}", "endDate": "{end}"}')
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Event families we track. `scale` = typical surprise size (fixed constants => no look-ahead).
# `risk_sign`: effect of a positive surprise on risk appetite under a "strong data = tighter policy"
# regime (-1 = risk-negative/hawkish). This is a modelling assumption: flip it if your regime differs.
EVENT_FAMILIES = {
    "CPI":          dict(pattern=r"\bCPI\b|Consumer Price",        scale=0.1,    risk_sign=-1, weight=1.0),
    "NFP":          dict(pattern=r"Non-?Farm|Nonfarm",             scale=50_000, risk_sign=-1, weight=1.0),
    "UNEMPLOYMENT": dict(pattern=r"Unemployment Rate",             scale=0.1,    risk_sign=+1, weight=0.7),
    "WAGES":        dict(pattern=r"Average Hourly Earnings",       scale=0.1,    risk_sign=-1, weight=0.5),
    "FOMC":         dict(pattern=r"FOMC|Federal Funds Rate",       scale=0.25,   risk_sign=-1, weight=1.0),
}
CONTEXT_FAMILIES = ("CPI", "NFP", "FOMC")   # families that get dedicated timing/surprise features
IMPACT_WEIGHT = {"high": 1.0, "medium": 0.5, "low": 0.2}
SENTIMENT_HALFLIFE_DAYS = 5.0
NO_EVENT_HOURS = 720.0   # cap for "hours since/to event" features

# Walk-forward settings
WF_TRAIN_DAYS = 90
WF_TEST_DAYS = 15
WF_STEP_DAYS = 15
WF_EMBARGO_DAYS = 1
TOP_K = 3
