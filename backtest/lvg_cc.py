"""Combined study: covered calls x leverage, with Sharpe ratios.

Questions answered (user):
  * why 2-3%/30d gross premium != +30%/yr  -> decomposition printed
  * CC + UPRO-like leverage combined       -> lever the CC equity curves
  * lever plain vs lever CC'd vs mix       -> full grid comparison
  * best L for returns / most stomachable  -> CAGR-max L and DD/Sharpe at L
  * Sharpe of all configs                  -> weekly excess over DGS10 path

Leverage model (same as div_vintage.ideal_leverage): weekly rebalanced,
lr_t = 1 + L*r_t - (L-1)*rf_t  with rf from the DGS10 path (borrow at the
10y proxy, consistent with earlier runs). Wipe-out check: any lr<=0 kills
the account (reported as CAGR=-100).

Configs (per vintage):
  base            no CC (div_vintage growth curve)
  cc_h0           g25_lowbeta|always, frictionless
  cc_h5           g25_lowbeta|always, 5% bid-ask haircut  (realistic)
  cc_all_h5       all|always, 5% haircut (max-income variant)
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cc_vintage2 import (VINTAGES, RF, END, load_book, run)  # noqa
from multi_vintage import stats  # noqa

L_GRID = [round(x, 2) for x in np.arange(1.0, 3.01, 0.05)]


def lever(ser, rf_w, L):
    r = ser.pct_change().dropna()
    rf = rf_w.reindex(r.index).ffill()
    lr = 1 + L * r - (L - 1) * rf
    if (lr <= 0).any():
        return None
    eq = ser.iloc[0] * lr.cumprod()
    eq = pd.concat([ser.iloc[:1], eq])
    return eq


def metrics(ser, rf_w, label):
    r = ser.pct_change().dropna()
    rf = rf_w.reindex(r.index).ffill()
    ex = r - rf
    sharpe = float(ex.mean() / ex.std() * math.sqrt(52)) if ex.std() > 0 \
        else 0.0
    yrs = (ser.index[-1] - ser.index[0]).days / 365.25
    cagr = (ser.iloc[-1] / ser.iloc[0]) ** (1 / yrs) - 1
    dd = float((ser / ser.cummax() - 1).min())
    return {"cagr_pct": round(100 * cagr, 2),
            "maxdd_pct": round(100 * dd, 1),
            "sharpe": round(sharpe, 3),
            "final": round(float(ser.iloc[-1]))}


def main():
    px = pd.read_csv(os.path.join(HERE, "weekly_unadj.csv"),
                     index_col=0, parse_dates=True) \
        .resample("W-FRI").last().loc[:END]
    dv = pd.read_csv(os.path.join(HERE, "weekly_divs.csv"),
                     index_col=0, parse_dates=True) \
        .resample("W-FRI").sum().loc[:END]
    sma = px.rolling(40).mean()
    ret = px.pct_change(fill_method=None)
    vol = ret.rolling(26).std() * math.sqrt(52)
    dgs10 = pd.read_csv(os.path.join(HERE, "dgs10.csv"),
                        index_col=0, parse_dates=True).iloc[:, 0]
    vix = pd.read_csv(os.path.join(HERE, "vix_weekly.csv"),
                      index_col=0, parse_dates=True).iloc[:, 0] / 100.0
    uplift = (vix.reindex(vol["^GSPC"].index) / vol["^GSPC"]).clip(
        0.8, 2.0).ffill()
    div_curves = pd.read_csv(os.path.join(HERE, "div_curves.csv"),
                             index_col=0, parse_dates=True)

    out = {}
    curves_out = {}
    for year, start in VINTAGES.items():
        book = load_book(year)
        dates = px.loc[start:END].index
        rfr = (dgs10.reindex(dates, method="ffill") / 100).fillna(RF[year])
        rf_w = rfr / 52.0
        gs = {t: v[1] for t, v in book.items()}
        betas = {t: v[2] for t, v in book.items()}
        med_b = sorted(betas.values())[len(betas) // 2]
        g25lb = {t for t in book if gs[t] <= .25 and betas[t] <= med_b}

        base = div_curves[f"{year}_gro"].dropna()
        cc_h0, _ = run(px, dv, sma, vol, uplift, rfr, dates, book,
                       RF[year], g25lb, "cc_h0", "always", 0.42, 0.0)
        cc_h5, _ = run(px, dv, sma, vol, uplift, rfr, dates, book,
                       RF[year], g25lb, "cc_h5", "always", 0.42, 0.05)
        ca_h5, _ = run(px, dv, sma, vol, uplift, rfr, dates, book,
                       RF[year], set(book), "cc_all_h5", "always", 0.42,
                       0.05)
        series = {"base": base, "cc_h0": cc_h0, "cc_h5": cc_h5,
                  "cc_all_h5": ca_h5}
        vout = {}
        for name, ser in series.items():
            m1 = metrics(ser, rf_w, name)
            grid = {}
            best_L, best_c = 1.0, m1["cagr_pct"]
            for L in L_GRID:
                lev = lever(ser, rf_w, L)
                if lev is None:
                    grid[str(L)] = None
                    continue
                mm = metrics(lev, rf_w, f"{name} {L}x")
                grid[str(L)] = mm
                if mm["cagr_pct"] > best_c:
                    best_c, best_L = mm["cagr_pct"], L
            vout[name] = {"L1": m1,
                          "L1_5": grid.get("1.5"), "L2": grid.get("2.0"),
                          "ideal_L": best_L,
                          "at_ideal": grid.get(str(best_L), m1)}
            curves_out[f"{year}_{name}"] = ser
        out[str(year)] = vout
        print(f"{year}: " + " ".join(
            f"{n}[1x {v['L1']['cagr_pct']:.1f}/S{v['L1']['sharpe']:.2f}"
            f" 1.5x {v['L1_5']['cagr_pct'] if v['L1_5'] else 'X'}"
            f" L*={v['ideal_L']}]" for n, v in vout.items()), flush=True)

    json.dump(out, open(os.path.join(HERE, "lvg_cc_results.json"), "w"),
              indent=1)
    pd.DataFrame(curves_out).to_csv(os.path.join(HERE, "lvg_cc_curves.csv"))


if __name__ == "__main__":
    main()
