"""DAY-BY-DAY covered-call simulation (user: "calculate what'd actually
happen day by day", "keep as much of the credit as possible").

Data: real daily unadjusted closes + real daily dividends, 1985-2026.
Options still need Black-Scholes (no listed-options history to 1990),
priced with IV = realized 126d vol x daily VIX/realized-SPX ratio
(data-derived, clipped 0.8-2.0).

Every trading day the engine:
  1. credits real dividends on shares held
  2. manages every open call:
       - expiry day: pay intrinsic (that's the real cash cost of letting
         it be exercised & buying shares back, or rolling at expiry),
         then immediately sell next month's call
       - roll80 mode: if delta > 0.80, buy back at model value, resell
       - tp50 mode: if call value has dropped to 50% of what we sold it
         for, buy it back early (lock the win), resell a fresh one
       - exp mode: never touch it before expiry (cheapest exit: only
         ever pay intrinsic, never pay remaining time value)
  3. opens calls on uncovered eligible stocks (always covered)
  4. runs the normal VMI tranche program (200d SMA support, under IV)
  5. reinvests the cash pool with discipline (support + under IV)
  6. marks equity net of open short-call value

Credit-retention levers tested (the "keep the 827M" request):
  - strike distance: delta 42 (Adam) vs 30 / 25 / 20 (further away =
    ITM less often = fewer give-backs, but smaller premium)
  - roll policy: roll80 (Adam) vs expiry-only vs take-profit-50%
  - universe: ideal (g<=25 + low beta) vs dow-like mature (g<=12)
    per Adam's "ideally Dow Jones 30 stocks" vs all 16
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
from multi_vintage import B2000_GRO, INITIAL, dcf_factor  # noqa

N16 = 16
CAP = INITIAL / N16
TRANCHE = CAP / 3.0
GAP_DAYS = 56                 # 8 weeks
END = "2026-07-18"
ND = NormalDist()
DTE = 35                      # calendar days
ROLL_DELTA = 0.80

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
    if sig <= 0 or S <= 0 or K <= 0 or T <= 1e-9:
        return max(S - K, 0.0)
    sq = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + sig * sig / 2) * T) / sq
    return S * ND.cdf(d1) - K * math.exp(-r * T) * ND.cdf(d1 - sq)


def call_delta(S, K, sig, r, T):
    if sig <= 0 or S <= 0 or K <= 0 or T <= 1e-9:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + sig * sig / 2) * T) / (sig * math.sqrt(T))
    return ND.cdf(d1)


def strike_for_delta(S, sig, r, T, delta):
    d1 = ND.inv_cdf(delta)
    return S * math.exp((r + sig * sig / 2) * T - d1 * sig * math.sqrt(T))


def run(D, book, rf, cc_set, tgt_delta, mode):
    """D: dict of prepared arrays. mode: 'roll80' | 'exp' | 'tp50'."""
    dates, px, dv, sma, sig_iv, rfd, exp_map, ti = (
        D["dates"], D["px"], D["dv"], D["sma"], D["sig"], D["rf"],
        D["exp_map"], D["ti"])
    nD = len(dates)
    anchor = dates[0]
    iv0, g_iv = {}, {}
    for t, (pe, g, b, m) in book.items():
        if t not in ti:
            continue
        col = px[:, ti[t]]
        ok = np.where(~np.isnan(col))[0]
        if not len(ok):
            continue
        iv0[t] = col[ok[0]] * dcf_factor(g, b, rf) * m / pe
        g_iv[t] = min(g, 0.10)
    iv_grow = {t: math.log(1 + g_iv[t]) / 365.25 for t in iv0}

    def iv_at(t, i):
        return iv0[t] * math.exp(iv_grow[t] * (dates[i] - anchor).days)

    sh = {t: 0.0 for t in iv0}
    tr, last_add = {}, {}
    cash, pool = INITIAL, 0.0
    oc = {}          # t -> [K, exp_i, cov, prem_ps]
    gross = back = 0.0
    n_otm = n_itm = n_roll = n_tp = n_open = 0
    prem_pct = 0.0
    eq = np.empty(nD)

    def sell(t, p, sig, r, i):
        nonlocal pool, gross, n_open, prem_pct
        e = exp_map[i]
        T = max((dates[min(e, nD - 1)] - dates[i]).days, 1) / 365.0
        K = strike_for_delta(p, sig, r, T, tgt_delta)
        pr = bs_call(p, K, sig, r, T)
        pool += pr * sh[t]
        gross += pr * sh[t]
        n_open += 1
        prem_pct += pr / p * (30.0 / max(T * 365, 1))
        oc[t] = [K, e, sh[t], pr]

    for i in range(nD):
        d = dates[i]
        r = rfd[i]
        # 1. dividends
        for t in sh:
            if sh[t] > 0 and t in ti:
                x = dv[i, ti[t]]
                if x > 0:
                    pool += sh[t] * x
        # 2. manage open calls
        for t in list(oc):
            K, e, cov, pr0 = oc[t]
            p = px[i, ti[t]]
            if np.isnan(p):
                if i >= e:
                    del oc[t]
                continue
            sig = sig_iv[i, ti[t]]
            if i >= e:                    # expiry: pay intrinsic only
                if p > K:
                    pool -= (p - K) * cov
                    back += (p - K) * cov
                    n_itm += 1
                else:
                    n_otm += 1
                del oc[t]
                if t in cc_set and sig > 0:
                    sell(t, p, sig, r, i)
                continue
            T = (dates[e if e < nD else nD - 1] - d).days / 365.0
            if mode == "roll80":
                if call_delta(p, K, sig, r, T) > ROLL_DELTA:
                    v = bs_call(p, K, sig, r, T) * cov
                    pool -= v; back += v; n_roll += 1
                    del oc[t]
                    if sig > 0:
                        sell(t, p, sig, r, i)
            elif mode == "tp50":
                v_ps = bs_call(p, K, sig, r, T)
                if v_ps <= 0.5 * pr0:
                    pool -= v_ps * cov; back += v_ps * cov; n_tp += 1
                    del oc[t]
                    if sig > 0:
                        sell(t, p, sig, r, i)
        # 3. open calls on uncovered eligible
        for t in cc_set:
            if t in oc or t not in ti or sh.get(t, 0) <= 0:
                continue
            p = px[i, ti[t]]
            sig = sig_iv[i, ti[t]]
            if np.isnan(p) or np.isnan(sig) or sig <= 0:
                continue
            sell(t, p, sig, r, i)
        # 4. tranche program
        for t in iv0:
            p = px[i, ti[t]]
            if np.isnan(p):
                continue
            if t not in tr:
                if p < iv_at(t, i) and cash >= TRANCHE:
                    sh[t] += TRANCHE / p
                    tr[t] = 1; last_add[t] = d; cash -= TRANCHE
            elif tr[t] < 3 and (d - last_add[t]).days >= GAP_DAYS:
                s = sma[i, ti[t]]
                if not np.isnan(s) and p <= s * 1.01 and p < iv_at(t, i) \
                        and cash >= TRANCHE:
                    sh[t] += TRANCHE / p; tr[t] += 1; last_add[t] = d
                    cash -= TRANCHE
        # 5. reinvest pool with discipline
        if pool > 0:
            best, bt, bp = 9e9, None, 0.0
            for t in iv0:
                p = px[i, ti[t]]
                s = sma[i, ti[t]]
                if np.isnan(p) or np.isnan(s):
                    continue
                ivd = iv_at(t, i)
                if p <= s * 1.01 and p < ivd and p / ivd < best:
                    best, bt, bp = p / ivd, t, p
            if bt is not None:
                sh[bt] += pool / bp
                pool = 0.0
        # 6. mark
        mv = cash + pool
        for t, s_ in sh.items():
            if s_ <= 0:
                continue
            p = px[i, ti[t]]
            if np.isnan(p):
                continue
            mv += s_ * p
            if t in oc:
                K, e, cov, _ = oc[t]
                T = (dates[e if e < nD else nD - 1] - d).days / 365.0
                sig = sig_iv[i, ti[t]]
                mv -= bs_call(p, K, sig if sig > 0 else 0.0, r, T) * cov
        eq[i] = mv

    ser = pd.Series(eq, index=dates)
    meta = {"gross": round(gross), "giveback": round(back),
            "net": round(gross - back),
            "kept_pct": round(100 * (gross - back) / gross, 1)
            if gross else 0.0,
            "n_open": n_open, "expired_worthless": n_otm,
            "expired_itm": n_itm, "rolls80": n_roll, "tp_closes": n_tp,
            "avg_prem_pct_30d": round(100 * prem_pct / n_open, 2)
            if n_open else 0.0}
    return ser, meta


def stats_of(ser, rf_ann):
    yrs = (ser.index[-1] - ser.index[0]).days / 365.25
    cagr = (ser.iloc[-1] / ser.iloc[0]) ** (1 / yrs) - 1
    dd = (ser / ser.cummax() - 1).min()
    w = ser.resample("W-FRI").last().dropna()
    rw = w.pct_change().dropna()
    ex = rw - rf_ann.reindex(rw.index).ffill() / 52
    sharpe = float(ex.mean() / ex.std() * math.sqrt(52)) if ex.std() else 0
    return round(cagr * 100, 2), round(dd * 100, 1), round(sharpe, 3)


def main():
    px_df = pd.read_csv(os.path.join(HERE, "daily_unadj.csv"),
                        index_col=0, parse_dates=True).loc[:END]
    dv_df = pd.read_csv(os.path.join(HERE, "daily_divs.csv"),
                        index_col=0, parse_dates=True) \
        .reindex(px_df.index).fillna(0.0)
    sma_df = px_df.rolling(200).mean()
    rv = px_df.pct_change(fill_method=None).rolling(126).std() \
        * math.sqrt(252)
    vix = px_df["^VIX"] / 100.0
    up = (vix / rv["^GSPC"]).clip(0.8, 2.0).ffill().fillna(1.0)
    sig_df = rv.mul(up, axis=0)
    dgs10 = pd.read_csv(os.path.join(HERE, "dgs10.csv"),
                        index_col=0, parse_dates=True).iloc[:, 0]
    rf_daily = (dgs10.reindex(px_df.index, method="ffill") / 100)

    out = {}
    curves = {}
    for year, start in VINTAGES.items():
        sub = px_df.loc[start:]
        dates = sub.index
        ti = {t: j for j, t in enumerate(px_df.columns)}
        arr = {
            "dates": dates,
            "px": sub.values,
            "dv": dv_df.loc[start:].values,
            "sma": sma_df.loc[start:].values,
            "sig": sig_df.loc[start:].values,
            "rf": rf_daily.loc[start:].fillna(RF[year]).values,
            "ti": ti,
            "exp_map": np.searchsorted(
                dates.values, dates.values + np.timedelta64(DTE, "D")),
        }
        book = load_book(year)
        gs = {t: v[1] for t, v in book.items()}
        betas = {t: v[2] for t, v in book.items()}
        med_b = sorted(betas.values())[len(betas) // 2]
        UNIV = {
            "ideal": {t for t in book
                      if gs[t] <= .25 and betas[t] <= med_b},
            "dow": {t for t in book if gs[t] <= .12},
            "all": set(book),
        }
        rfa = rf_daily.loc[start:].fillna(RF[year])
        vout = {}
        ser0, _ = run(arr, book, RF[year], set(), 0.42, "exp")
        c, dd, sp = stats_of(ser0, rfa)
        vout["none"] = {"cagr": c, "dd": dd, "sharpe": sp,
                        "final": round(ser0.iloc[-1])}
        curves[f"{year}_none"] = ser0.resample("W-FRI").last()

        cfgs = []
        for u in ("ideal", "dow"):
            cfgs += [(u, .42, "roll80"), (u, .42, "exp"), (u, .42, "tp50"),
                     (u, .30, "exp"), (u, .25, "exp"), (u, .20, "exp"),
                     (u, .30, "tp50")]
        cfgs += [("all", .42, "roll80"), ("all", .30, "exp"),
                 ("all", .20, "exp")]
        for u, tdel, mode in cfgs:
            name = f"{u}|d{int(tdel*100)}|{mode}"
            ser, meta = run(arr, book, RF[year], UNIV[u], tdel, mode)
            c, dd, sp = stats_of(ser, rfa)
            vout[name] = {"cagr": c, "dd": dd, "sharpe": sp,
                          "final": round(ser.iloc[-1]),
                          "n_stocks": len(UNIV[u]), **meta}
            if name == "ideal|d30|exp":
                curves[f"{year}_ideal_d30"] = ser.resample("W-FRI").last()
        out[str(year)] = vout
        b = vout["none"]["cagr"]
        top = sorted(((v["cagr"] - b, n) for n, v in vout.items()
                      if n != "none"), reverse=True)[:4]
        print(f"{year} base {b}%: " +
              " ".join(f"{n}:{u:+.2f}" for u, n in top), flush=True)

    json.dump(out, open(os.path.join(HERE, "cc_daily_results.json"), "w"),
              indent=1)
    pd.DataFrame(curves).to_csv(os.path.join(HERE, "cc_daily_curves.csv"))


if __name__ == "__main__":
    main()
