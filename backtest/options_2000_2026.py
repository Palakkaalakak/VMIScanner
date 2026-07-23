#!/usr/bin/env python3
"""
Structural test: extend the 2000-2013 options overlays to 2026.

Runs the SAME variants (plain-derived CC, PMCC full / half-pyramid /
convert with growing 6.25% spend cap) on B2000_DEF and B2000_GRO, but
over the FULL window 2000-01-07 -> 2026 data end. If the overlay only
worked in the sideways 2000-2013 market, the 2013-2026 bull leg will
expose it.

Outputs:
  eq26_{book}_{variant}.csv          full-window curves
  stats_options_2000_2026.json       headline stats + sub-period split
                                     (2000-2013 vs 2013-2026 CAGR)
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cc_daily3 import RF, prepare, arrays_for, run as run_cc     # noqa: E402
from pmcc_daily import run_pmcc                                  # noqa: E402
from multi_vintage import B2000_DEF, B2000_GRO                   # noqa: E402

START = "2000-01-07"
SPLIT = "2013-12-27"
INITIAL = 1_000_000.0
BOOKS = {"defensive": B2000_DEF, "growth": B2000_GRO}


def full_stats(ser):
    yrs = (ser.index[-1] - ser.index[0]).days / 365.25
    cagr = (ser.iloc[-1] / ser.iloc[0]) ** (1 / yrs) - 1
    dd = (ser / ser.cummax() - 1).min()
    # sub-period CAGRs
    a = ser.loc[:SPLIT]
    b = ser.loc[SPLIT:]
    y1 = (a.index[-1] - a.index[0]).days / 365.25
    y2 = (b.index[-1] - b.index[0]).days / 365.25
    c1 = (a.iloc[-1] / a.iloc[0]) ** (1 / y1) - 1
    c2 = (b.iloc[-1] / b.iloc[0]) ** (1 / y2) - 1
    return {"final": round(float(ser.iloc[-1])),
            "cagr_pct": round(100 * cagr, 2),
            "max_dd_pct": round(100 * dd, 1),
            "cagr_2000_2013": round(100 * c1, 2),
            "cagr_2013_2026": round(100 * c2, 2)}


def main():
    px_df, dv_df, sma_df, sig_df, mo12, rf_daily = prepare()
    D = arrays_for(2000, START, px_df, dv_df, sma_df, sig_df, mo12, rf_daily)
    rf = RF[2000]

    stats = {}
    for key, book in BOOKS.items():
        cset = {t for t, (pe, g, b, m) in book.items() if g <= 0.15}
        stats[key] = {"cc_names": sorted(cset)}

        # covered calls
        ser, _ = run_cc(D, book, rf, cset, initial=INITIAL)
        ser.to_frame("equity").to_csv(os.path.join(HERE, f"eq26_{key}_cc.csv"))
        stats[key]["cc"] = full_stats(ser)
        print(f"{key} cc        CAGR {stats[key]['cc']['cagr_pct']:6.2f}% "
              f"(13: {stats[key]['cc']['cagr_2000_2013']:.1f} / "
              f"26: {stats[key]['cc']['cagr_2013_2026']:.1f}) "
              f"DD {stats[key]['cc']['max_dd_pct']:.1f}%", flush=True)

        # buy & hold baseline (no options) via run_cc with empty cc_set
        ser, _ = run_cc(D, book, rf, set(), initial=INITIAL)
        ser.to_frame("equity").to_csv(
            os.path.join(HERE, f"eq26_{key}_plain.csv"))
        stats[key]["plain"] = full_stats(ser)
        print(f"{key} plain     CAGR {stats[key]['plain']['cagr_pct']:6.2f}%",
              flush=True)

        for vname, kw in (("pmcc", dict(full_tranche=True, spend_cap=True)),
                          ("pmcc_hp", dict(full_tranche=True, spend_cap=True,
                                           half_pyramid=True)),
                          ("pmcc_conv", dict(full_tranche=True,
                                             spend_cap=True, convert=True))):
            ser, meta = run_pmcc(D, book, rf, style="ds1", pmcc_set=cset,
                                 initial=INITIAL, **kw)
            ser.to_frame("equity").to_csv(
                os.path.join(HERE, f"eq26_{key}_{vname}.csv"))
            stats[key][vname] = full_stats(ser)
            s = stats[key][vname]
            print(f"{key} {vname:9s} CAGR {s['cagr_pct']:6.2f}% "
                  f"(13: {s['cagr_2000_2013']:.1f} / "
                  f"26: {s['cagr_2013_2026']:.1f}) DD {s['max_dd_pct']:.1f}%",
                  flush=True)
        # incremental save so resets don't lose finished books
        with open(os.path.join(HERE, "stats_options_2000_2026.json"),
                  "w") as f:
            json.dump(stats, f, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
