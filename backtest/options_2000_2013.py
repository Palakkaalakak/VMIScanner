#!/usr/bin/env python3
"""
Options overlays for the 2000-2013 dashboard backtest ($1M books).

1. Threshold sensitivity: covered calls with the g-cutoff swept from 10% to
   'everything' — is g<=15% a real structural edge or a lucky number?
2. PMCC (natural leverage, ds1 style) on BOTH books, three flavors:
   nat (roll = full pyramid), halfpyr (roll keeps same share count,
   profit banked), conv (convert to shares when long call is deep ITM).
   Allocation rules identical to stock version: 6.25% cap in PREMIUM
   DOLLARS SPENT, 3 tranches, 56-day gap, IV-gated entries.

Outputs:
  eq_{book}_pmcc.csv / eq_{book}_pmcc_hp.csv / eq_{book}_pmcc_conv.csv
  cc_threshold_sweep.json
  stats_options_2000_2013.json   (headline stats for every variant)
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
END_CUT = "2013-12-27"
INITIAL = 1_000_000.0
BOOKS = {"defensive": B2000_DEF, "growth": B2000_GRO}


def hstats(ser):
    ser = ser.loc[:END_CUT]
    yrs = (ser.index[-1] - ser.index[0]).days / 365.25
    cagr = (ser.iloc[-1] / ser.iloc[0]) ** (1 / yrs) - 1
    dd = (ser / ser.cummax() - 1).min()
    return ser, {"final": round(float(ser.iloc[-1])),
                 "cagr_pct": round(100 * cagr, 2),
                 "max_dd_pct": round(100 * dd, 1)}


def main():
    px_df, dv_df, sma_df, sig_df, mo12, rf_daily = prepare()
    D = arrays_for(2000, START, px_df, dv_df, sma_df, sig_df, mo12, rf_daily)
    rf = RF[2000]

    # ---------- 1. threshold sweep (skip if already saved) ----------
    SKIP_SWEEP = os.path.exists(os.path.join(HERE, "cc_threshold_sweep.json"))
    sweep = {}
    for key, book in ({} if SKIP_SWEEP else BOOKS).items():
        sweep[key] = {}
        for thr in (0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.25, 9.99):
            cset = {t for t, (pe, g, b, m) in book.items() if g <= thr}
            ser, meta = run_cc(D, book, rf, cset, initial=INITIAL)
            _, s = hstats(ser)
            s["n_cc"] = len(cset)
            sweep[key][f"{thr:.2f}"] = s
            print(f"sweep {key} g<={thr:.2f} n={len(cset):2d} "
                  f"CAGR {s['cagr_pct']:6.2f}%  DD {s['max_dd_pct']:6.1f}%",
                  flush=True)
    if not SKIP_SWEEP:
        with open(os.path.join(HERE, "cc_threshold_sweep.json"), "w") as f:
            json.dump(sweep, f, indent=2)

    # ---------- 2. PMCC on both books ----------
    stats = {}
    # spend_cap=True: allocation rule enforced on PREMIUM DOLLARS SPENT --
    # cumulative new capital per name capped at 6.25% of initial, exactly
    # like the stock book (exposure may exceed that; spend may not).
    variants = [("pmcc",      dict(full_tranche=True, spend_cap=True)),
                ("pmcc_hp",   dict(full_tranche=True, half_pyramid=True,
                                   spend_cap=True)),
                ("pmcc_conv", dict(full_tranche=True, convert=True,
                                   spend_cap=True))]
    for key, book in BOOKS.items():
        cset = {t for t, (pe, g, b, m) in book.items() if g <= 0.15}
        stats[key] = {"cc_names": sorted(cset),
                      "plain_names": sorted(set(book) - cset)}
        for vname, kw in variants:
            ser, meta = run_pmcc(D, book, rf, style="ds1", pmcc_set=cset,
                                 initial=INITIAL, **kw)
            ser, s = hstats(ser)
            ser.to_frame("equity").to_csv(
                os.path.join(HERE, f"eq_{key}_{vname}.csv"))
            s["meta"] = {k: meta[k] for k in
                         ("short_net", "long_net_decay", "dividends",
                          "long_rolls", "converts") if k in meta}
            stats[key][vname] = s
            print(f"{key} {vname:9s} final ${s['final']:>12,}  "
                  f"CAGR {s['cagr_pct']:6.2f}%  DD {s['max_dd_pct']:6.1f}%",
                  flush=True)

    with open(os.path.join(HERE, "stats_options_2000_2013.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
