"""UPRO-STYLE leverage on the daily equity curves (user correction:
leveraged-ETF mechanics, NOT margin borrowing).

How a UPRO-like fund works and how it's simulated here:
  * DAILY RESET: each day the fund delivers L x (that day's return),
    re-balancing internally. You can never lose more than 100% and
    there are NO margin calls -- the fund does the resetting.
  * COSTS: the fund carries financing on the extra (L-1) exposure at
    short-term rates plus an expense ratio. UPRO charges 0.91%/yr;
    we model fee = 0.95%/yr (era-documented, conservative) and
    financing = DGS10 path (same rate series used everywhere else).
  * VOLATILITY DECAY: the well-known cost of daily reset -- captured
    automatically by compounding daily.

    r_fund(day) = L * r_portfolio(day) - (L-1) * rf(day)/252 - fee/252

Applied to: none (no options), cc_always (calls on everything),
wheel -- daily curves from cc_daily2. Grid L = 1.0 .. 3.0 step 0.25.
For each: CAGR, worst peak-to-trough drop, Sharpe.
"""
import json
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FEE = 0.0095
YEARS = ["1990", "1995", "2000", "2005", "2010", "2015", "2020"]


def stats(ser, rf_d):
    yrs = (ser.index[-1] - ser.index[0]).days / 365.25
    cagr = (ser.iloc[-1] / ser.iloc[0]) ** (1 / yrs) - 1
    dd = (ser / ser.cummax() - 1).min()
    r = ser.pct_change().dropna()
    ex = r - rf_d.reindex(r.index).ffill() / 252
    sh = float(ex.mean() / ex.std() * math.sqrt(252)) if ex.std() else 0
    return round(cagr * 100, 2), round(dd * 100, 1), round(sh, 3)


def main():
    curves = pd.read_csv(os.path.join(HERE, "cc_daily2_curves.csv"),
                         index_col=0, parse_dates=True)
    dgs10 = pd.read_csv(os.path.join(HERE, "dgs10.csv"),
                        index_col=0, parse_dates=True).iloc[:, 0]
    rf_d = (dgs10.reindex(curves.index, method="ffill") / 100).ffill()

    out = {}
    for y in YEARS:
        vout = {}
        for cfg in ("none", "cc_always", "wheel"):
            col = f"{y}_{cfg}"
            if col not in curves.columns:
                continue
            ser = curves[col].dropna()
            r = ser.pct_change().fillna(0.0)
            rw = rf_d.reindex(ser.index).ffill().fillna(0.0)
            grid = {}
            for L in np.arange(1.0, 3.001, 0.25):
                rl = L * r - (L - 1) * rw / 252 - (FEE / 252 if L > 1
                                                   else 0.0)
                lc = ser.iloc[0] * (1 + rl).cumprod()
                if lc.min() <= 0:
                    continue
                c, dd, sh = stats(lc, rf_d)
                grid[f"{L:.2f}"] = {"cagr": c, "dd": dd, "sharpe": sh}
            vout[cfg] = grid
        out[y] = vout
        n2 = vout["none"].get("2.00", {})
        c2 = vout["cc_always"].get("2.00", {})
        print(f"{y}: none 2x -> {n2.get('cagr')}% dd {n2.get('dd')}% | "
              f"cc 2x -> {c2.get('cagr')}% dd {c2.get('dd')}%", flush=True)

    json.dump(out, open(os.path.join(HERE, "upro_results.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
