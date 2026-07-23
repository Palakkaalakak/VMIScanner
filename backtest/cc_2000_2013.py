#!/usr/bin/env python3
"""
Covered-call variant of the 2000-2013 dashboard backtest.

Re-runs the two $1M books (B2000_DEF defensive, B2000_GRO growth from
multi_vintage.py) through the day-by-day covered-call engine (cc_daily3.run),
selling monthly ~delta-0.42 calls ONLY on the CC-viable names (book growth
g <= 15%). Fast growers are held as plain shares with no calls.

Outputs (consumed by scanner/backtest_tab.py):
  eq_defensive_cc.csv   equity curve, truncated to 2013-12-27
  eq_growth_cc.csv
  stats_cc_2000_2013.json   final value / CAGR / max DD + which names got calls

NOTE on consistency: the original dashboard curves (eq_defensive.csv etc.)
were simulated on adjusted prices (dividends implicitly reinvested).
This CC engine uses unadjusted prices + explicit cash dividends. Same
economics, slightly different plumbing; the caption in the UI discloses it.
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cc_daily3 import RF, prepare, arrays_for, run as run_cc  # noqa: E402
from multi_vintage import B2000_DEF, B2000_GRO               # noqa: E402

START = "2000-01-07"
END_CUT = "2013-12-27"
INITIAL = 1_000_000.0


def main():
    px_df, dv_df, sma_df, sig_df, mo12, rf_daily = prepare()
    D = arrays_for(2000, START, px_df, dv_df, sma_df, sig_df, mo12, rf_daily)

    stats = {}
    for key, book in (("defensive", B2000_DEF), ("growth", B2000_GRO)):
        cc_set = {t for t, (pe, g, b, m) in book.items() if g <= 0.15}
        missing = [t for t in book if t not in D["ti"]]
        if missing:
            print(f"WARNING {key}: missing price data for {missing}")
        ser, meta = run_cc(D, book, RF[2000], cc_set, initial=INITIAL)
        ser = ser.loc[:END_CUT]
        out = os.path.join(HERE, f"eq_{key}_cc.csv")
        ser.to_frame("equity").to_csv(out)
        yrs = (ser.index[-1] - ser.index[0]).days / 365.25
        cagr = (ser.iloc[-1] / ser.iloc[0]) ** (1 / yrs) - 1
        dd = (ser / ser.cummax() - 1).min()
        stats[key] = {
            "final": round(float(ser.iloc[-1])),
            "cagr_pct": round(100 * cagr, 2),
            "max_dd_pct": round(100 * dd, 1),
            "cc_names": sorted(cc_set),
            "plain_names": sorted(set(book) - cc_set),
            "call_net": meta["call_net"],
            "dividends": meta["dividends"],
            "calls_total": meta["calls_total"],
        }
        print(f"{key}: final ${ser.iloc[-1]:,.0f}  CAGR {100*cagr:.2f}%  "
              f"DD {100*dd:.1f}%  calls on {len(cc_set)}/{len(book)} names")

    with open(os.path.join(HERE, "stats_cc_2000_2013.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("saved eq_*_cc.csv + stats_cc_2000_2013.json")


if __name__ == "__main__":
    main()
