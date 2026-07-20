"""Patch 5y-weekly betas (vs ^GSPC, ending at vintage date) into
vintage_inputs_<YEAR>.json — retry-safe against yf batch flakiness."""
import json
import sys
import time

import numpy as np
import yfinance as yf

for year, vd in ((2020, "2020-01-02"), (2015, "2015-01-02")):
    p = f"backtest/vintage_inputs_{year}.json"
    d = json.load(open(p))
    tickers = [c["ticker"].replace(".", "-") for c in d["candidates"]]
    px = None
    for attempt in range(4):
        px = yf.download(tickers + ["^GSPC"], start=f"{year-5}-01-01", end=vd,
                         interval="1wk", auto_adjust=True, progress=False)["Close"]
        if "^GSPC" in px.columns and px["^GSPC"].notna().sum() > 100:
            break
        print(f"[{year}] batch retry {attempt+1}", flush=True)
        time.sleep(15)
    mkt = px["^GSPC"].pct_change(fill_method=None).dropna()
    n = 0
    for c in d["candidates"]:
        t = c["ticker"].replace(".", "-")
        if t not in px.columns:
            continue
        r = px[t].pct_change(fill_method=None).dropna()
        j = r.index.intersection(mkt.index)
        if len(j) >= 100:
            rr, mm = r.loc[j], mkt.loc[j]
            c["beta"] = round(float(np.cov(rr, mm)[0, 1] / np.var(mm)), 2)
            n += 1
    json.dump(d, open(p, "w"), indent=1)
    full = [c for c in d["candidates"] if c["trailing_pe"] and c["growth"] is not None
            and c["beta"] is not None and c["ocf_mult"]]
    print(year, "betas fixed:", n, "full candidates:", len(full), flush=True)
