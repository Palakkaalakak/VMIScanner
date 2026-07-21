"""Fetch weekly prices from 1985 for deep-vintage books (per-ticker, reliable).

Union of 1990/1995/2005/2010 hand-built era book tickers + replacements
+ benchmarks. yfinance data cliff is 1985-01-01 for most legacy tickers.
"""
import os
import time

import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "weekly_deep.csv")

TICKERS = [
    # 1990 book
    "HD", "MCD", "DIS", "NKE", "WMT", "KO", "PEP", "SYY",
    "JNJ", "ABT", "MDT", "ADP", "GWW", "EMR", "ITW", "AXP",
    # 1995 additions
    "MSFT", "INTC", "ORCL", "UNH",
    # 2005 additions
    "QCOM", "SYK", "BDX", "LOW", "MO", "DHR", "UPS",
    # 2010 additions
    "AAPL", "GOOGL", "TJX", "ROST", "ISRG", "UNP", "FAST",
    # scandal replacements pool
    "GIS", "CL", "CHD",
    # benchmarks
    "SPY", "^GSPC",
]

START = "1985-01-01"


def main():
    if os.path.exists(OUT):
        done = pd.read_csv(OUT, index_col=0, parse_dates=True)
        cols = set(done.columns)
    else:
        done, cols = None, set()
    frames = {} if done is None else {c: done[c] for c in done.columns}
    for t in TICKERS:
        if t in cols:
            continue
        for attempt in range(3):
            try:
                h = yf.Ticker(t).history(start=START, interval="1wk",
                                         auto_adjust=True)
                if len(h) >= 100:
                    s = h["Close"].copy()
                    s.index = pd.to_datetime(s.index).tz_localize(None)
                    s.index = s.index.normalize()
                    frames[t] = s
                    print(f"{t:8} {len(h):5} rows  {s.index[0].date()} -> "
                          f"{s.index[-1].date()}")
                    break
                print(f"{t:8} SHORT ({len(h)}) attempt {attempt+1}")
            except Exception as e:
                print(f"{t:8} ERR {e} attempt {attempt+1}")
            time.sleep(2)
        time.sleep(0.4)
    df = pd.DataFrame(frames).sort_index()
    df.to_csv(OUT)
    print(f"\nwrote {OUT}: {df.shape[1]} tickers x {df.shape[0]} weeks "
          f"({df.index[0].date()} -> {df.index[-1].date()})")
    missing = [t for t in TICKERS if t not in df.columns]
    if missing:
        print("MISSING:", missing)


if __name__ == "__main__":
    main()
