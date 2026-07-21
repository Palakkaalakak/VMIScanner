"""Covered-call overlay v2 -- accurate simulation of Adam's PDF mechanics.

Improvements over cc_vintage.py (all from 'Selling Covered Calls 2024' PDF
+ data-derived calibration, no invented numbers):

1. IMPLIED-VOL PREMIUMS: options are priced at IV, not realized vol.
   Calibration: weekly ^VIX / 26w-realized-vol(^GSPC), 1990-2026
   (median 1.27, stable by decade: 1.32/1.25/1.23/1.21). We apply the
   *time-varying* weekly ratio (clipped 0.8-2.0) to each stock's realized
   vol: sigma_iv(t,d) = RV_stock(t,d) * VIX(d)/RV_spx(d).

2. ROLL MECHANICS (PDF Exit Scenarios):
   * Scenario 1  (OTM at expiry): option expires worthless, keep premium,
     immediately sell-to-open a new call the same week.
   * Scenario 2  (ITM ~5 days before expiry): buy-to-close at BS value,
     sell-to-open new delta-42 35DTE call at the same time.
   * Scenario 3  (delta > 80 mid-cycle): roll immediately (buy back,
     sell new delta-42 35DTE).
   Old model settled cash = intrinsic at expiry and then waited; new
   model keeps the position continuously covered and collects the new
   extrinsic on every roll.

3. ENTRY GATE (PDF: "overextended and/or below a level of resistance"):
   configurable -- gate='sma'   : fresh opens only when p > 40w SMA
                                  (rolls always allowed, per scenarios 2/3)
                   gate='always': continuously covered (Adam re-sells at
                                  every expiry per Scenario 1)

4. MEASUREMENT: achieved premium %/30d per open is logged so we can
   compare with Adam's 2-3%/30d target and the user's 1% worst case.

Mark-to-market subtracts full BS value of open short calls (not just
intrinsic) for accuracy.
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
CYCLE = 5                      # weeks (~35 days)
T_YR = 35.0 / 365.0
ROLL_DELTA = 0.80              # PDF scenario 3
UPLIFT_LO, UPLIFT_HI = 0.8, 2.0

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
    if sig <= 0 or S <= 0 or K <= 0 or T <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + sig * sig / 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return S * ND.cdf(d1) - K * math.exp(-r * T) * ND.cdf(d2)


def call_delta(S, K, sig, r, T):
    if sig <= 0 or S <= 0 or K <= 0 or T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + sig * sig / 2) * T) / (sig * math.sqrt(T))
    return ND.cdf(d1)


def strike_for_delta(S, sig, r, T, delta):
    d1 = ND.inv_cdf(delta)
    return S * math.exp((r + sig * sig / 2) * T - d1 * sig * math.sqrt(T))


def run(px, dv, sma, vol, uplift, rfr, dates, book, rf, cc_set, label,
        gate="sma", tgt_delta=0.42, haircut=0.0):
    # haircut: transaction friction -- sell at (1-h)*BS, buy back at
    # (1+h)*BS (bid-ask spread sensitivity, reported as a range)
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
    open_cc = {}                          # t -> [strike, expiry_i, shares]
    prem_tot, buyback_tot = 0.0, 0.0
    n_opens, n_rolls, prem_pct_sum = 0, 0, 0.0
    eq = []

    def sigma_iv(t, d):
        rv = vol.at[d, t] if t in vol.columns else np.nan
        u = uplift.at[d] if d in uplift.index else np.nan
        if np.isnan(rv) or rv <= 0:
            return np.nan
        return rv * (u if not np.isnan(u) else 1.0)

    def sell_new(t, p, sig, r_now, i):
        nonlocal pool, prem_tot, n_opens, prem_pct_sum
        K = strike_for_delta(p, sig, r_now, T_YR, tgt_delta)
        prem_ps = bs_call(p, K, sig, r_now, T_YR) * (1.0 - haircut)
        pool += prem_ps * sh[t]
        prem_tot += prem_ps * sh[t]
        n_opens += 1
        prem_pct_sum += (prem_ps / p) * (30.0 / 35.0)   # normalize to /30d
        open_cc[t] = [K, i + CYCLE, sh[t]]

    for i, d in enumerate(dates):
        # dividends
        for t, s_ in sh.items():
            dps = dv.at[d, t] if t in dv.columns else 0.0
            if dps and not np.isnan(dps):
                pool += s_ * dps

        # --- covered-call management (PDF exit scenarios) ---
        for t in list(open_cc):
            K, exp_i, cov = open_cc[t]
            p = px.at[d, t] if t in px.columns else np.nan
            if np.isnan(p):
                if i >= exp_i:
                    del open_cc[t]
                continue
            sig = sigma_iv(t, d)
            r_now = rfr.at[d] if d in rfr.index else rf
            wk_left = exp_i - i
            T_rem = max(wk_left, 0) * 7.0 / 365.0
            delta = call_delta(p, K, sig, r_now, T_rem) \
                if not np.isnan(sig) else (1.0 if p > K else 0.0)
            if i >= exp_i:
                # expiry week: OTM -> worthless; ITM -> settle intrinsic
                if p > K:
                    pool -= (p - K) * cov
                    buyback_tot += (p - K) * cov
                del open_cc[t]
                # Scenario 1: immediately sell new call (gate-dependent)
                if t in cc_set and not np.isnan(sig) and sig > 0:
                    s40 = sma.at[d, t]
                    if gate == "always" or (not np.isnan(s40) and p > s40):
                        sell_new(t, p, sig, r_now, i)
            elif delta > ROLL_DELTA or (wk_left <= 1 and p > K):
                # Scenario 3 (deep ITM) or Scenario 2 (ITM near expiry):
                # buy-to-close at BS value, sell-to-open new call now
                bb = bs_call(p, K, sig if not np.isnan(sig) else 0.0,
                             r_now, T_rem) * cov * (1.0 + haircut)
                pool -= bb
                buyback_tot += bb
                del open_cc[t]
                n_rolls += 1
                if not np.isnan(sig) and sig > 0:
                    sell_new(t, p, sig, r_now, i)

        # fresh opens on uncovered cc_set members
        for t in cc_set:
            if t in open_cc or t not in sh or t not in px.columns:
                continue
            p, s40 = px.at[d, t], sma.at[d, t]
            sig = sigma_iv(t, d)
            if np.isnan(p) or np.isnan(sig) or sig <= 0:
                continue
            if gate == "sma" and (np.isnan(s40) or p <= s40):
                continue
            r_now = rfr.at[d] if d in rfr.index else rf
            sell_new(t, p, sig, r_now, i)

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

        # disciplined reinvestment of pool (divs + net CC income)
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

        # mark-to-market: subtract BS value of open short calls
        mv = cash + pool
        r_now = rfr.at[d] if d in rfr.index else rf
        for t, s_ in sh.items():
            p = px.at[d, t]
            if np.isnan(p):
                continue
            mv += s_ * p
            if t in open_cc:
                K, exp_i, cov = open_cc[t]
                sig = sigma_iv(t, d)
                T_rem = max(exp_i - i, 0) * 7.0 / 365.0
                mv -= bs_call(p, K, sig if not np.isnan(sig) else 0.0,
                              r_now, T_rem) * cov
        eq.append(mv)

    ser = pd.Series(eq, index=dates, name=label)
    meta = {"prem_collected": round(prem_tot),
            "buyback_paid": round(buyback_tot),
            "net_cc_income": round(prem_tot - buyback_tot),
            "n_opens": n_opens, "n_rolls": n_rolls,
            "avg_prem_pct_30d": round(100 * prem_pct_sum / n_opens, 3)
            if n_opens else 0.0}
    return ser, meta


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
    # IV uplift: VIX / realized 26w vol of ^GSPC, clipped (data-derived)
    vix = pd.read_csv(os.path.join(HERE, "vix_weekly.csv"),
                      index_col=0, parse_dates=True).iloc[:, 0] / 100.0
    rv_spx = vol["^GSPC"]
    uplift = (vix.reindex(rv_spx.index) / rv_spx).clip(
        UPLIFT_LO, UPLIFT_HI).ffill()

    old = json.load(open(os.path.join(HERE, "cc_results.json")))
    base = json.load(open(os.path.join(HERE, "div_results.json")))

    out = {}
    curves = {}
    for year, start in VINTAGES.items():
        book = load_book(year)
        dates = px.loc[start:END].index
        rfr = (dgs10.reindex(dates, method="ffill") / 100).fillna(RF[year])
        gs = {t: v[1] for t, v in book.items()}
        betas = {t: v[2] for t, v in book.items()}
        med_b = sorted(betas.values())[len(betas) // 2]
        lowbeta8 = set(sorted(betas, key=lambda t: betas[t])[:8])
        g25lb = {t for t in book if gs[t] <= .25 and betas[t] <= med_b}
        g20lb = {t for t in book if gs[t] <= .20 and betas[t] <= med_b}
        configs = [
            ("all|sma", set(book), "sma", 0.42),
            ("all|always", set(book), "always", 0.42),
            ("g<=25|sma", {t for t, g in gs.items() if g <= .25}, "sma", .42),
            ("g25_lowbeta|sma", g25lb, "sma", 0.42),
            ("g25_lowbeta|always", g25lb, "always", 0.42),
            ("g25_lowbeta|always_d45", g25lb, "always", 0.45),
            ("g20_lowbeta|always", g20lb, "always", 0.42),
            ("lowbeta_half|always", lowbeta8, "always", 0.42),
        ]
        frictions = [("g25_lowbeta|always_h5", g25lb, "always", 0.42, .05),
                     ("g25_lowbeta|always_h10", g25lb, "always", 0.42, .10),
                     ("all|always_h5", set(book), "always", 0.42, .05),
                     ("all|always_h10", set(book), "always", 0.42, .10)]
        vout = {"none": {
            "cagr_pct": base[str(year)]["growth"]["cagr_pct"],
            "final": base[str(year)]["growth"]["final"]}}
        for name, cset, gate, tdel, *h in configs + frictions:
            hc = h[0] if h else 0.0
            ser, meta = run(px, dv, sma, vol, uplift, rfr, dates, book,
                            RF[year], cset, f"{year} {name}", gate, tdel,
                            haircut=hc)
            st = stats(ser, f"{year} {name}")
            vout[name] = {"cagr_pct": st["cagr_pct"], "final": st["final"],
                          "maxdd_pct": st["max_drawdown_pct"],
                          "n_cc_stocks": len(cset), **meta}
            if name == "g25_lowbeta|always":
                curves[f"{year}"] = ser
        # old-model comparator
        vout["OLD_g25_lowbeta"] = {
            "cagr_pct": old[str(year)]["g25_lowbeta"]["cagr_pct"],
            "final": old[str(year)]["g25_lowbeta"]["final"]}
        out[str(year)] = vout
        b = vout["none"]["cagr_pct"]
        row = " ".join(f"{n}:{v['cagr_pct'] - b:+.2f}"
                       for n, v in vout.items() if n != "none")
        print(f"{year} (base {b:.2f}%): {row}", flush=True)

    json.dump(out, open(os.path.join(HERE, "cc2_results.json"), "w"),
              indent=1)
    pd.DataFrame(curves).to_csv(os.path.join(HERE, "cc2_curves.csv"))


if __name__ == "__main__":
    main()
