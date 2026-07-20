"""Build one weekly adjusted-close price store for the multi-vintage backtest.

Per-ticker yf.Ticker().history() (reliable — batch download intermittently
returns truncated 26-week frames). Union of 2020+2015 candidates + SPY +
^GSPC, 2009-06 -> today (runway for 40w SMA + 5y beta window before 2015).
Cached to weekly_vintage.csv; resumable.
"""
import json
import os
import time

import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "weekly_vintage.csv")
START = "2009-06-01"


def wanted_tickers():
    ts = {"SPY", "^GSPC"}
    for y in (2020, 2015):
        d = json.load(open(os.path.join(HERE, f"vintage_inputs_{y}.json")))
        ts.update(c["ticker"].replace(".", "-") for c in d["candidates"])
    return sorted(ts)


def main():
    tickers = wanted_tickers()
    if os.path.exists(OUT):
        store = pd.read_csv(OUT, index_col=0, parse_dates=True)
    else:
        store = pd.DataFrame()
    todo = [t for t in tickers if t not in store.columns]
    print(f"{len(tickers)} tickers wanted, {len(todo)} to fetch", flush=True)
    fails = []
    for k, t in enumerate(todo):
        ok = False
        for attempt in range(3):
            try:
                h = yf.Ticker(t).history(start=START, interval="1wk",
                                         auto_adjust=True)
                if h is not None and len(h) >= 200:
                    s = h["Close"]
                    s.index = s.index.tz_localize(None)
                    store[t] = s
                    ok = True
                    break
            except Exception:
                pass
            time.sleep(5)
        if not ok:
            fails.append(t)
        if (k + 1) % 20 == 0:
            store.to_csv(OUT)
            print(f"{k+1}/{len(todo)} fetched ({len(fails)} fails)", flush=True)
        time.sleep(0.2)
    store.sort_index(inplace=True)
    store.to_csv(OUT)
    print(f"DONE: {len(store.columns)} tickers, {store.index[0].date()} -> "
          f"{store.index[-1].date()}; fails: {fails}", flush=True)


if __name__ == "__main__":
    main()
