"""Fetch unadjusted weekly closes + dividend series for all book tickers.

Explicit dividend reinvestment needs:
  * unadjusted prices (auto_adjust=False Close) so dividends aren't
    double-counted (our current stores are dividend-adjusted), and
  * per-week dividend cash per share.
Per-ticker fetch (batch download is unreliable). Resumable.
"""
import json
import os
import time

import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
PX_OUT = os.path.join(HERE, "weekly_unadj.csv")
DV_OUT = os.path.join(HERE, "weekly_divs.csv")
START = "1985-01-01"


def book_tickers():
    ts = set()
    for y in (1990, 1995, 2005, 2010, 2015, 2020):
        d = json.load(open(os.path.join(HERE, f"books_growth_{y}.json")))
        ts |= set(d["growth_book"])
    # 2000 book from multi_vintage
    import sys
    sys.path.insert(0, HERE)
    from multi_vintage import B2000_GRO
    ts |= set(B2000_GRO)
    ts |= {"CL", "GIS", "CHD", "GPC", "SYK"}  # repl pool (kept for reference)
    ts |= {"SPY", "^GSPC"}
    return sorted(t.replace(".", "-") for t in ts)


def main():
    tickers = book_tickers()
    if os.path.exists(PX_OUT):
        px = pd.read_csv(PX_OUT, index_col=0, parse_dates=True)
        dv = pd.read_csv(DV_OUT, index_col=0, parse_dates=True)
        pxf = {c: px[c] for c in px.columns}
        dvf = {c: dv[c] for c in dv.columns}
    else:
        pxf, dvf = {}, {}
    for t in tickers:
        if t in pxf:
            continue
        for attempt in range(3):
            try:
                h = yf.Ticker(t).history(start=START, interval="1wk",
                                         auto_adjust=False)
                if len(h) >= 100:
                    idx = pd.to_datetime(h.index).tz_localize(None).normalize()
                    p = h["Close"].copy(); p.index = idx
                    d = h["Dividends"].copy(); d.index = idx
                    pxf[t], dvf[t] = p, d
                    print(f"{t:8} {len(h):5} rows {idx[0].date()} -> "
                          f"{idx[-1].date()} divs>0: {(d > 0).sum()}")
                    break
                print(f"{t:8} SHORT {len(h)} attempt {attempt+1}")
            except Exception as e:
                print(f"{t:8} ERR {e} attempt {attempt+1}")
            time.sleep(2)
        time.sleep(0.4)
    pd.DataFrame(pxf).sort_index().to_csv(PX_OUT)
    pd.DataFrame(dvf).sort_index().to_csv(DV_OUT)
    px = pd.DataFrame(pxf)
    print(f"\nwrote {px.shape[1]} tickers x {px.shape[0]} weeks")
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        print("MISSING:", missing)


if __name__ == "__main__":
    main()
