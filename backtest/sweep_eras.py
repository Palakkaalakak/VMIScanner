#!/usr/bin/env python3
"""CC g-threshold sweep across ALL THREE eras (dow books, $100k) to test
whether a growth cutoff for call-selling is structural or era luck."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cc_daily3 import (RF, VINTAGES, prepare, arrays_for,  # noqa: E402
                       run as run_cc, load_books3)


def main():
    px, dv, sma, sig, mo12, rfd = prepare()
    books = load_books3()
    out = {}
    for year, start in VINTAGES.items():
        D = arrays_for(year, start, px, dv, sma, sig, mo12, rfd)
        book = books[year]["dow"]
        out[str(year)] = {}
        for thr in (0.10, 0.15, 0.20, 0.25, 9.99):
            cset = {t for t, (pe, g, b, m) in book.items() if g <= thr}
            ser, _ = run_cc(D, book, RF[year], cset)
            yrs = (ser.index[-1] - ser.index[0]).days / 365.25
            cagr = (ser.iloc[-1] / ser.iloc[0]) ** (1 / yrs) - 1
            dd = (ser / ser.cummax() - 1).min()
            out[str(year)][f"{thr:.2f}"] = {
                "n_cc": len(cset), "cagr_pct": round(100 * cagr, 2),
                "max_dd_pct": round(100 * dd, 1)}
            print(f"{year} g<={thr:.2f} n={len(cset):2d} "
                  f"CAGR {100*cagr:6.2f}% DD {100*dd:6.1f}%", flush=True)
    json.dump(out, open(os.path.join(HERE, "cc_sweep_eras.json"), "w"),
              indent=2)
    print("saved cc_sweep_eras.json")


if __name__ == "__main__":
    main()
