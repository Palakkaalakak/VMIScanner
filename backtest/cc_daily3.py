"""Day-by-day covered-call engine v3 — the 'which stocks deserve calls' study.

User request: $100k start (not $1M); vintages 1990/2000/2015; two books each
(dow = era selection re-run with Dow candidates; claude = my PEG-rule picks);
find the STOCK-TYPE criteria for covered calls (Adam lost huge upside selling
calls on NVDA/MSFT-type rockets); and raise premium RETENTION (later profit-
taking thresholds, different roll triggers) without hurting total return.

Stock-type criteria tested (all era-knowable, no hindsight):
  all       calls on every book stock (baseline winner so far)
  lowg15    only stocks with book growth <= 15%  (skip the rockets)
  lowg20    only stocks with book growth <= 20%
  lowbeta   only stocks with beta <= 1.0 (calm movers)
  slowmo    dynamic: skip any stock whose trailing 12m price gain > 30%
            (recomputed daily -- NVDA-type behaviour, not a label)
  overval   dynamic: write only while price >= intrinsic value (upside
            above IV is 'borrowed' anyway -- cap it; keep upside below IV)

Retention levers (on the best stock sets):
  tp65/tp80/tp90  buy the call back once 65/80/90% of its premium has
                  decayed away (profit-take), then re-sell; higher threshold
                  = wait longer = keep more per call
  roll70/roll90   roll the call up-and-out when delta hits 0.70 / 0.90
                  (baseline 0.80); rolling later = fewer, bigger buybacks

Engine mechanics identical to cc_daily2.py (real daily closes 1985-2026,
real dividend dates, Black-Scholes with IV = 126d realized vol x daily
VIX/SPX-RV ratio, delta-42 35-DTE calls, intrinsic paid at expiry, VMI
3-tranche buying, premiums reinvested into cheapest below-IV book stock).
INITIAL = $100,000 (fractional contract coverage, so CAGRs are scale-free;
dollar outputs reflect the $100k start).
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
from multi_vintage import dcf_factor  # noqa: E402

INITIAL = 100_000.0
N16 = 16
GAP_DAYS = 56
END = "2026-07-18"
ND = NormalDist()
DTE = 35

VINTAGES = {1990: "1990-01-05", 2000: "2000-01-07", 2015: "2015-01-09"}
RF = {1990: .0794, 2000: .065, 2015: .0212}


def load_books3():
    d = json.load(open(os.path.join(HERE, "books3.json")))
    out = {}
    for y, bb in d.items():
        out[int(y)] = {name: {t.replace(".", "-"):
                              (v["pe"], v["g"], v["beta"], v["ocf_mult"])
                              for t, v in bk.items()}
                       for name, bk in bb.items()}
    return out


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


def k_for_d1(S, sig, r, T, d1):
    return S * math.exp((r + sig * sig / 2) * T - d1 * sig * math.sqrt(T))


def run(D, book, rf, cc_set, tgt_delta=0.42, gate="always",
        roll_delta=0.80, tp=None, slow_mo=False, initial=INITIAL,
        hot_delta=None, hot_g=0.20, hot_mo=None):
    """gate: 'always' | 'overval' (p>=IV) ; tp: capture fraction or None.
    hot_delta: if set, stocks with book growth > hot_g (or, if hot_mo is
    set, trailing-12m price gain > hot_mo, recomputed daily) get calls at
    this smaller delta (further OTM = more room to run) instead of being
    skipped."""
    dates, px, dv, sma, sig_m, rfd, exp_map, ti, mo12 = (
        D["dates"], D["px"], D["dv"], D["sma"], D["sig"], D["rf"],
        D["exp_map"], D["ti"], D["mo12"])
    nD = len(dates)
    cap = initial / N16
    tranche = cap / 3.0
    d1c = ND.inv_cdf(tgt_delta)
    d1_hot = ND.inv_cdf(hot_delta) if hot_delta else None
    hot_static = {t for t, v in book.items() if v[1] > hot_g} \
        if (hot_delta and hot_mo is None) else set()
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
    cash, pool = initial, 0.0
    oc = {}    # t -> [K, exp_i, cov, prem_total]
    c_gross = c_back = 0.0
    cc_win = cc_loss = 0
    eq = np.empty(nD)

    def can_write(t, i, p):
        if t not in cc_set:
            return False
        if slow_mo and not np.isnan(mo12[i, ti[t]]) and mo12[i, ti[t]] > 0.30:
            return False
        if gate == "overval" and p < iv_at(t, i):
            return False
        return True

    def sell_call(t, p, sig, r, i):
        nonlocal pool, c_gross
        e = exp_map[i]
        T = max((dates[min(e, nD - 1)] - dates[i]).days, 1) / 365.0
        dd1 = d1c
        if d1_hot is not None:
            if hot_mo is not None:
                if not np.isnan(mo12[i, ti[t]]) and mo12[i, ti[t]] > hot_mo:
                    dd1 = d1_hot
            elif t in hot_static:
                dd1 = d1_hot
        K = k_for_d1(p, sig, r, T, dd1)
        pr = bs_call(p, K, sig, r, T)
        pool += pr * sh[t]
        c_gross += pr * sh[t]
        oc[t] = [K, e, sh[t], pr * sh[t]]

    def close_call(t, cost):
        nonlocal pool, c_back, cc_win, cc_loss
        prem = oc[t][3]
        pool -= cost
        c_back += cost
        if cost > prem:
            cc_loss += 1
        else:
            cc_win += 1
        del oc[t]

    for i in range(nD):
        d = dates[i]
        r = rfd[i]
        for t in sh:
            if sh[t] > 0 and t in ti:
                x = dv[i, ti[t]]
                if x > 0:
                    pool += sh[t] * x

        # ---- manage open calls ----
        for t in list(oc):
            K, e, cov, pr = oc[t]
            p = px[i, ti[t]]
            if np.isnan(p):
                if i >= e:
                    del oc[t]
                continue
            sig = sig_m[i, ti[t]]
            if i >= e:                                   # expiry
                close_call(t, (p - K) * cov if p > K else 0.0)
                if sig > 0 and can_write(t, i, p):
                    sell_call(t, p, sig, r, i)
                continue
            T = (dates[e if e < nD else nD - 1] - d).days / 365.0
            val = bs_call(p, K, sig, r, T)
            if call_delta(p, K, sig, r, T) > roll_delta:   # roll trigger
                close_call(t, val * cov)
                if sig > 0 and can_write(t, i, p):
                    sell_call(t, p, sig, r, i)
                continue
            if tp is not None and val * cov <= (1 - tp) * pr:  # profit-take
                close_call(t, val * cov)
                if sig > 0 and can_write(t, i, p):
                    sell_call(t, p, sig, r, i)

        # ---- open new calls ----
        for t in cc_set:
            if t in oc or t not in ti or sh.get(t, 0) <= 0:
                continue
            p = px[i, ti[t]]
            sig = sig_m[i, ti[t]]
            if np.isnan(p) or np.isnan(sig) or sig <= 0:
                continue
            if can_write(t, i, p):
                sell_call(t, p, sig, r, i)

        # ---- tranche program ----
        for t in iv0:
            p = px[i, ti[t]]
            if np.isnan(p):
                continue
            ntr = tr.get(t, 0)
            if ntr >= 3:
                continue
            trigger = False
            if ntr == 0:
                trigger = p < iv_at(t, i)
            elif (d - last_add[t]).days >= GAP_DAYS:
                s = sma[i, ti[t]]
                trigger = (not np.isnan(s) and p <= s * 1.01
                           and p < iv_at(t, i))
            if not trigger or cash < tranche:
                continue
            sh[t] = sh.get(t, 0.0) + tranche / p
            cash -= tranche
            tr[t] = ntr + 1
            last_add[t] = d

        # ---- reinvest pool ----
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

        # ---- mark ----
        mv = cash + pool
        for t, s_ in sh.items():
            if s_ <= 0:
                continue
            p = px[i, ti[t]]
            if np.isnan(p):
                continue
            mv += s_ * p
        for t, (K, e, cov, _) in oc.items():
            p = px[i, ti[t]]
            if np.isnan(p):
                continue
            T = (dates[e if e < nD else nD - 1] - d).days / 365.0
            mv -= bs_call(p, K, sig_m[i, ti[t]], r, T) * cov
        eq[i] = mv

    ser = pd.Series(eq, index=dates)
    tot = cc_win + cc_loss
    meta = {"call_gross": round(c_gross), "call_back": round(c_back),
            "call_net": round(c_gross - c_back),
            "kept_pct": round(100 * (c_gross - c_back) / c_gross, 1)
            if c_gross else 0.0,
            "calls_total": tot, "calls_profitable": cc_win,
            "calls_unprofitable": cc_loss,
            "unprofitable_pct": round(100 * cc_loss / tot, 1) if tot else 0.0}
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


def prepare():
    px_df = pd.read_csv(os.path.join(HERE, "daily_unadj.csv"),
                        index_col=0, parse_dates=True).loc[:END] \
        .dropna(how="all")
    px_df = px_df.ffill(limit=5)
    dv_df = pd.read_csv(os.path.join(HERE, "daily_divs.csv"),
                        index_col=0, parse_dates=True) \
        .reindex(index=px_df.index, columns=px_df.columns).fillna(0.0)
    sma_df = px_df.rolling(200).mean()
    rv = px_df.pct_change(fill_method=None).rolling(126).std() \
        * math.sqrt(252)
    up = (px_df["^VIX"] / 100.0 / rv["^GSPC"]).clip(0.8, 2.0) \
        .ffill().fillna(1.0)
    sig_df = rv.mul(up, axis=0)
    mo12 = px_df.pct_change(252, fill_method=None)
    dgs10 = pd.read_csv(os.path.join(HERE, "dgs10.csv"),
                        index_col=0, parse_dates=True).iloc[:, 0]
    rf_daily = dgs10.reindex(px_df.index, method="ffill") / 100
    return px_df, dv_df, sma_df, sig_df, mo12, rf_daily


def arrays_for(year, start, px_df, dv_df, sma_df, sig_df, mo12, rf_daily):
    sub = px_df.loc[start:]
    dates = sub.index
    ti = {t: j for j, t in enumerate(px_df.columns)}
    return {
        "dates": dates, "px": sub.values,
        "dv": dv_df.loc[start:].values,
        "sma": sma_df.loc[start:].values,
        "sig": sig_df.loc[start:].values,
        "rf": rf_daily.loc[start:].fillna(RF[year]).values,
        "ti": ti, "mo12": mo12.loc[start:].values,
        "exp_map": np.searchsorted(
            dates.values, dates.values + np.timedelta64(DTE, "D")),
    }


def main():
    px_df, dv_df, sma_df, sig_df, mo12, rf_daily = prepare()
    books = load_books3()
    out, curves = {}, {}
    for year, start in VINTAGES.items():
        arr = arrays_for(year, start, px_df, dv_df, sma_df, sig_df,
                         mo12, rf_daily)
        rfa = rf_daily.loc[start:].fillna(RF[year])
        for bname, book in books[year].items():
            allset = set(book)
            lowg15 = {t for t, v in book.items() if v[1] <= 0.15}
            lowg20 = {t for t, v in book.items() if v[1] <= 0.20}
            lowbeta = {t for t, v in book.items() if v[2] <= 1.0}
            # (cc_set, gate, roll_delta, tp, slow_mo)
            cfgs = [
                ("none",       set(),   "always", .80, None, False),
                ("all",        allset,  "always", .80, None, False),
                ("lowg15",     lowg15,  "always", .80, None, False),
                ("lowg20",     lowg20,  "always", .80, None, False),
                ("lowbeta",    lowbeta, "always", .80, None, False),
                ("slowmo",     allset,  "always", .80, None, True),
                ("overval",    allset,  "overval", .80, None, False),
                # retention levers on the 'all' set:
                ("all_tp65",   allset,  "always", .80, .65, False),
                ("all_tp80",   allset,  "always", .80, .80, False),
                ("all_tp90",   allset,  "always", .80, .90, False),
                ("all_roll70", allset,  "always", .70, None, False),
                ("all_roll90", allset,  "always", .90, None, False),
                ("all_tp80_roll90", allset, "always", .90, .80, False),
            ]
            # hot-stock treatment: don't skip the rockets, write FURTHER
            # OTM on them (smaller delta = more room to run)
            hot_cfgs = [
                # (name, hot_delta, hot_g, hot_mo, roll, tp)
                ("hot25g20",    .25, .20, None, .80, None),
                ("hot15g20",    .15, .20, None, .80, None),
                ("hot25mo30",   .25, .20, .30,  .80, None),
                ("hot25g20_tp90", .25, .20, None, .80, .90),
            ]
            vout = {}
            for name, cset, gate, rd, tp, smo in cfgs:
                ser, meta = run(arr, book, RF[year], cset, .42, gate,
                                rd, tp, smo)
                c, dd, sp = stats_of(ser, rfa)
                vout[name] = {"cagr": c, "dd": dd, "sharpe": sp,
                              "final": round(ser.iloc[-1]), **meta}
                if name in ("none", "all"):
                    curves[f"{year}_{bname}_{name}"] = ser
            for name, hd, hg, hm, rd, tp in hot_cfgs:
                ser, meta = run(arr, book, RF[year], allset, .42, "always",
                                rd, tp, False, INITIAL, hd, hg, hm)
                c, dd, sp = stats_of(ser, rfa)
                vout[name] = {"cagr": c, "dd": dd, "sharpe": sp,
                              "final": round(ser.iloc[-1]), **meta}
            out[f"{year}_{bname}"] = vout
            b = vout["none"]["cagr"]
            print(f"{year} {bname:6} base {b}%: " + " ".join(
                f"{n}:{v['cagr'] - b:+.2f}" for n, v in vout.items()
                if n != "none"), flush=True)

    json.dump(out, open(os.path.join(HERE, "cc_daily3_results.json"), "w"),
              indent=1)
    pd.DataFrame(curves).to_csv(os.path.join(HERE, "cc_daily3_curves.csv"))


if __name__ == "__main__":
    main()
