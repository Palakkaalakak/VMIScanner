"""Covered-call overlay on the no-sell, dividend-reinvest 7-vintage run.

Adam's rules (Lesson 12 + Selling Covered Calls 2024):
  * every ~30-45 days sell 1 call per 100 shares on suitable stocks
    (fundamentally strong, slower movers; avoid biotech/pharma)
  * strike at delta 40-45, 30-45 DTE; target 1.5-3% / 30 days
  * sell when the stock is over-extended / NOT at a buy point
  * if ITM near expiry: roll (buy back intrinsic, sell new) -> economics
    equal cash-settling max(P_T - K, 0) and keeping the shares
  * premium is income: goes to the disciplined reinvestment pool
    (buys cheapest-vs-IV book stock on support & under IV)

Pricing (no options history back to 1990): Black-Scholes at delta 0.42,
35 DTE, sigma = trailing 26-week realized vol (annualized), r = era DGS10
path. Realized vol <= implied vol in general, so premiums are conservative.
Timing gate per Adam: only open a call when price > 40w SMA (over-
extended), never at support where we'd want the upside.

Configs tested per vintage:
  none          (baseline = div_vintage run)
  all           CC on every holding
  g<=8/10/12/15/20  CC only on stocks with era expected growth <= X%
  lowbeta_half  CC on the 8 lowest-beta stocks ("other variable" probe)
"""
import json
import math
import os
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from multi_vintage import B2000_GRO, INITIAL, dcf_factor, stats  # noqa

N16 = 16
CAP = INITIAL / N16
TRANCHE = CAP / 3.0
GAP = 8
END = "2026-07-18"
ND = NormalDist()
D1_TARGET = ND.inv_cdf(0.42)      # delta-0.42 call
CYCLE = 5                          # weeks (~35 days)
T_YR = 35.0 / 365.0

VINTAGES = {
    1990: "1990-01-05", 1995: "1995-01-06", 2000: "2000-01-07",
    2005: "2005-01-07", 2010: "2010-01-08", 2015: "2015-01-09",
    2020: "2020-01-10",
}
RF = {1990: .0794, 1995: .0788, 2000: .065, 2005: .0423,
      2010: .0385, 2015: .0212, 2020: .0188}


def load_book(year):
    if year == 2000:
        return {t: v for t, v in B2000_GRO.items()}
    d = json.load(open(os.path.join(HERE, f"books_growth_{year}.json")))
    return {t.replace(".", "-"): (v["pe"], v["g"], v["beta"], v["ocf_mult"])
            for t, v in d["growth_book"].items()}


def bs_call(S, K, sig, r, T):
    if sig <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + sig * sig / 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return S * ND.cdf(d1) - K * math.exp(-r * T) * ND.cdf(d2)


def strike_for_delta(S, sig, r, T):
    # N(d1) = 0.42  ->  K = S * exp((r + sig^2/2)T - d1*sig*sqrt(T))
    return S * math.exp((r + sig * sig / 2) * T
                        - D1_TARGET * sig * math.sqrt(T))


def run(px, dv, sma, vol, rfr, dates, book, rf, cc_set, label):
    anchor = dates[0]
    iv0, g_iv = {}, {}
    for t, (pe, g, b, m) in book.items():
        s = px[t].loc[anchor:].dropna()
        if s.empty:
            continue
        iv0[t] = s.iloc[0] * dcf_factor(g, b, rf) * m / pe
        g_iv[t] = min(g, 0.10)

    def iv_at(t, d):
        return iv0[t] * (1 + g_iv[t]) ** ((d - anchor).days / 365.25)

    sh, tr, last = {}, {}, {}
    cash, pool = INITIAL, 0.0            # pool = divs + net CC income
    open_cc = {}                          # t -> (strike, expiry_i, shares)
    prem_tot, settle_tot = 0.0, 0.0
    eq = []
    for i, d in enumerate(dates):
        # dividends
        for t, s_ in sh.items():
            dps = dv.at[d, t] if t in dv.columns else 0.0
            if dps and not np.isnan(dps):
                pool += s_ * dps
        # covered calls: settle expiring, then (re)open
        for t in list(open_cc):
            K, exp_i, cov = open_cc[t]
            if i >= exp_i:
                p = px.at[d, t]
                if not np.isnan(p) and p > K:
                    cost = (p - K) * cov
                    pool -= cost
                    settle_tot += cost
                del open_cc[t]
        for t in cc_set:
            if t in open_cc or t not in sh or t not in px.columns:
                continue
            p, s40 = px.at[d, t], sma.at[d, t]
            sig = vol.at[d, t] if t in vol.columns else np.nan
            if np.isnan(p) or np.isnan(s40) or np.isnan(sig) or sig <= 0:
                continue
            if p <= s40:          # only when over-extended, never at support
                continue
            r_now = rfr.at[d] if d in rfr.index else rf
            K = strike_for_delta(p, sig, r_now, T_YR)
            prem = bs_call(p, K, sig, r_now, T_YR) * sh[t]
            pool += prem
            prem_tot += prem
            open_cc[t] = (K, i + CYCLE, sh[t])
        # tranche program
        for t in book:
            if t not in iv0 or t not in px.columns:
                continue
            p = px.at[d, t]
            if np.isnan(p):
                continue
            if t not in tr:
                if p < iv_at(t, d) and cash >= TRANCHE:
                    sh[t] = sh.get(t, 0) + TRANCHE / p
                    tr[t] = 1; last[t] = i; cash -= TRANCHE
            elif tr[t] < 3 and i - last[t] >= GAP:
                s = sma.at[d, t]
                if not np.isnan(s) and p <= s * 1.01 and p < iv_at(t, d) \
                        and cash >= TRANCHE:
                    sh[t] += TRANCHE / p; tr[t] += 1; last[t] = i
                    cash -= TRANCHE
        # disciplined reinvestment of pool (divs + CC income)
        if pool > 0:
            cands = []
            for t in book:
                if t not in iv0 or t not in px.columns:
                    continue
                p, s = px.at[d, t], sma.at[d, t]
                if np.isnan(p) or np.isnan(s):
                    continue
                ivd = iv_at(t, d)
                if p <= s * 1.01 and p < ivd:
                    cands.append((p / ivd, t, p))
            if cands:
                cands.sort()
                _, t, p = cands[0]
                sh[t] = sh.get(t, 0) + pool / p
                pool = 0.0
        # mark: subtract open short-call intrinsic
        mv = cash + pool
        for t, s_ in sh.items():
            p = px.at[d, t]
            if np.isnan(p):
                continue
            mv += s_ * p
            if t in open_cc:
                K, _, cov = open_cc[t]
                if p > K:
                    mv -= (p - K) * cov
        eq.append(mv)
    ser = pd.Series(eq, index=dates, name=label)
    return ser, round(prem_tot), round(settle_tot)


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

    out = {}
    curves = {}
    for year, start in VINTAGES.items():
        book = load_book(year)
        dates = px.loc[start:END].index
        rfr = (dgs10.reindex(dates, method="ffill") / 100).fillna(RF[year])
        gs = {t: v[1] for t, v in book.items()}
        betas = {t: v[2] for t, v in book.items()}
        lowbeta8 = set(sorted(betas, key=lambda t: betas[t])[:8])
        configs = {"all": set(book)}
        for x in (8, 10, 12, 15, 20, 25, 30):
            configs[f"g<={x}"] = {t for t, g in gs.items() if g <= x / 100}
        configs["lowbeta_half"] = lowbeta8
        top4g = set(sorted(gs, key=lambda t: -gs[t])[:4])
        top2g = set(sorted(gs, key=lambda t: -gs[t])[:2])
        configs["ex_top4_growth"] = set(book) - top4g
        configs["ex_top2_growth"] = set(book) - top2g
        vout = {}
        for name, cset in configs.items():
            ser, prem, settle = run(px, dv, sma, vol, rfr, dates, book,
                                    RF[year], cset, f"{year} {name}")
            st = stats(ser, f"{year} {name}")
            vout[name] = {"cagr_pct": st["cagr_pct"], "final": st["final"],
                          "maxdd_pct": st["max_drawdown_pct"],
                          "n_cc_stocks": len(cset),
                          "prem_collected": prem, "settle_paid": settle,
                          "net_cc_income": prem - settle}
            if name == "all":
                curves[f"{year}_all"] = ser
        out[str(year)] = vout
        base = json.load(open(os.path.join(HERE, "div_results.json")))
        vout["none"] = {"cagr_pct": base[str(year)]["growth"]["cagr_pct"],
                        "final": base[str(year)]["growth"]["final"],
                        "maxdd_pct":
                            base[str(year)]["growth"]["max_drawdown_pct"],
                        "n_cc_stocks": 0, "prem_collected": 0,
                        "settle_paid": 0, "net_cc_income": 0}
        row = " ".join(f"{n}:{v['cagr_pct']:.2f}%" for n, v in vout.items())
        print(f"{year}: {row}")

    json.dump(out, open(os.path.join(HERE, "cc_results.json"), "w"),
              indent=1)
    pd.DataFrame(curves).to_csv(os.path.join(HERE, "cc_curves.csv"))


if __name__ == "__main__":
    main()
