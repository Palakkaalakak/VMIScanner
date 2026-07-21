"""Leverage x Covered-Call combination study + Sharpe ratios.

Questions answered (user):
  * combine CC overlay with UPRO-like leverage?
  * better to leverage-only, CC-only, or mix?
  * which L maximizes CAGR; which is 'stomachable'?
  * Sharpe of every config.

Method: take weekly equity curves of
    base          (div_vintage no-CC growth run)
    cc_ideal      (g25_lowbeta|always, frictionless)
    cc_ideal_h5   (g25_lowbeta|always, 5% bid-ask haircut)
    cc_all        (all|always frictionless)
    cc_all_h5     (all|always, 5% haircut)
then apply daily-reset style weekly leverage:
    r_L = L * r  -  (L-1) * rf_w      (borrow at DGS10 path, as before)
Grid L in [1, 3] step 0.05.  For each: CAGR, maxDD, Sharpe
(weekly excess over DGS10/52, annualized sqrt(52)).
Report per vintage: unlevered stats, L*_cagr, L at maxDD<=55% ('stomach'),
and cross-config comparison at matched drawdown.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cc_vintage2 as cc2  # noqa

VINTAGES = cc2.VINTAGES
RF = cc2.RF
END = cc2.END


def sharpe(ser, rf_w):
    r = ser.pct_change(fill_method=None).dropna()
    ex = r - rf_w.reindex(r.index).fillna(method="ffill")
    if ex.std() == 0:
        return 0.0
    return float(ex.mean() / ex.std() * math.sqrt(52))


def maxdd(ser):
    return float((ser / ser.cummax() - 1).min() * 100)


def cagr(ser):
    yrs = (ser.index[-1] - ser.index[0]).days / 365.25
    return float(((ser.iloc[-1] / ser.iloc[0]) ** (1 / yrs) - 1) * 100)


def lever(ser, rf_w, L):
    r = ser.pct_change(fill_method=None).fillna(0.0)
    rw = rf_w.reindex(r.index).fillna(method="ffill").fillna(0.0)
    rl = L * r - (L - 1) * rw
    if (rl <= -1).any():
        return None
    return ser.iloc[0] * (1 + rl).cumprod()


def lvg_scan(ser, rf_w, dd_cap=-55.0):
    best = {"L": 1.0, "cagr": cagr(ser)}
    stomach = {"L": 1.0, "cagr": cagr(ser), "dd": maxdd(ser)}
    for L in np.arange(1.05, 3.001, 0.05):
        lc = lever(ser, rf_w, round(L, 2))
        if lc is None:
            break
        c, d = cagr(lc), maxdd(lc)
        if c > best["cagr"]:
            best = {"L": round(L, 2), "cagr": c, "dd": d,
                    "sharpe": sharpe(lc, rf_w)}
        if d >= dd_cap and c > stomach["cagr"]:
            stomach = {"L": round(L, 2), "cagr": c, "dd": d,
                       "sharpe": sharpe(lc, rf_w)}
    if "dd" not in best:
        best["dd"] = maxdd(ser)
        best["sharpe"] = sharpe(ser, rf_w)
    return best, stomach


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
        cc2.UPLIFT_LO, cc2.UPLIFT_HI).ffill()
    base_curves = pd.read_csv(os.path.join(HERE, "div_curves.csv"),
                              index_col=0, parse_dates=True)

    out = {}
    curves_store = {}
    for year, start in VINTAGES.items():
        book = cc2.load_book(year)
        dates = px.loc[start:END].index
        rfr = (dgs10.reindex(dates, method="ffill") / 100).fillna(RF[year])
        rf_w = rfr / 52.0
        gs = {t: v[1] for t, v in book.items()}
        betas = {t: v[2] for t, v in book.items()}
        med_b = sorted(betas.values())[len(betas) // 2]
        g25lb = {t for t in book if gs[t] <= .25 and betas[t] <= med_b}

        curves = {"base": base_curves[f"{year}_gro"].dropna()}
        specs = [("cc_ideal", g25lb, 0.0), ("cc_ideal_h5", g25lb, .05),
                 ("cc_all", set(book), 0.0), ("cc_all_h5", set(book), .05)]
        for name, cset, hc in specs:
            ser, _ = cc2.run(px, dv, sma, vol, uplift, rfr, dates, book,
                             RF[year], cset, name, gate="always",
                             tgt_delta=0.42, haircut=hc)
            curves[name] = ser
        vout = {}
        for name, ser in curves.items():
            ser = ser.dropna()
            un = {"cagr": round(cagr(ser), 2), "dd": round(maxdd(ser), 1),
                  "sharpe": round(sharpe(ser, rf_w), 3)}
            best, stomach = lvg_scan(ser, rf_w)
            vout[name] = {
                "unlevered": un,
                "L_cagr_max": {k: round(v, 2) if isinstance(v, float)
                               else v for k, v in best.items()},
                "L_dd55": {k: round(v, 2) if isinstance(v, float)
                           else v for k, v in stomach.items()},
            }
            if name in ("base", "cc_ideal_h5"):
                curves_store[f"{year}_{name}"] = ser
        out[str(year)] = vout
        print(f"{year}: " + " | ".join(
            f"{n}: {v['unlevered']['cagr']}% S{v['unlevered']['sharpe']}"
            f" -> L{v['L_dd55']['L']}: {v['L_dd55']['cagr']}%"
            for n, v in vout.items()), flush=True)

    json.dump(out, open(os.path.join(HERE, "cc_lvg_results.json"), "w"),
              indent=1)
    pd.DataFrame(curves_store).to_csv(
        os.path.join(HERE, "cc_lvg_curves.csv"))


if __name__ == "__main__":
    main()
