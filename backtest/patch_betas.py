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
    # chunked download: big batches silently return empty frames
    import pandas as pd
    frames = []
    todo = tickers + ["^GSPC"]
    for i in range(0, len(todo), 25):
        chunk = todo[i:i + 25]
        for attempt in range(3):
            f = yf.download(chunk, start=f"{year-5}-01-01", end=vd,
                            interval="1wk", auto_adjust=True,
                            progress=False)["Close"]
            if isinstance(f, pd.Series):
                f = f.to_frame(chunk[0])
            if len(f) and f.notna().any().any():
                frames.append(f)
                break
            print(f"[{year}] chunk {i} retry {attempt+1}", flush=True)
            time.sleep(10)
    px = pd.concat(frames, axis=1)
    px = px.loc[:, ~px.columns.duplicated()]
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
