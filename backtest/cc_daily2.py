"""Day-by-day engine v2: covered calls + cash-secured puts + the Wheel.

New vs cc_daily.py (user requests):
  * PER-CALL P&L: every call closed is scored premium-received minus
    buyback-paid.  A call closed at a net loss counts as UNPROFITABLE
    (user's definition, ignoring the share gain that offsets it).
  * BETTER ENTRIES for selling calls (tested, data-derived):
      always   sell whenever uncovered (old behaviour)
      ext      only when price > 200d SMA (Adam: over-extended)
      extnh    over-extended AND not at a fresh 52-week high
               (breakouts have no resistance above -> calls get run over;
                'below a level of resistance' per Adam)
  * BETTER STOCKS day-by-day: skip_hot excludes the 2 hottest book
    stocks (highest trailing 26-week return, recomputed daily, no
    hindsight) -- these are the ones that blow through strikes.
  * CASH-SECURED PUTS (Selling Cash Secured Puts 2024): when a tranche
    entry triggers (support + under IV), instead of buying shares, sell
    a 35-DTE delta-30 put at that entry; cash stays reserved.
      assigned  -> own the tranche at strike minus premium (discount!)
      expires   -> keep premium, re-arm the entry
  * THE WHEEL (Cashflow Wheel): puts for entries + covered calls on
    holdings.  VMI never sells, so our wheel is put -> hold -> calls
    (the sell-100-shares spoke is replaced by rolling, per no-sell rule).

Pricing: Black-Scholes, IV = 126d realized vol x daily VIX/SPX-RV ratio
(clipped 0.8-2.0, data-derived).  Real daily closes and real dividend
dates, 1985-2026.
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
GAP_DAYS = 56
END = "2026-07-18"
ND = NormalDist()
DTE = 35
ROLL_DELTA = 0.80
D1_CALL = None  # set per call from tgt_delta
D1_PUT = ND.inv_cdf(0.70)      # put delta -0.30 -> N(d1)=0.70

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


def bs_put(S, K, sig, r, T):
    if sig <= 0 or S <= 0 or K <= 0 or T <= 1e-9:
        return max(K - S, 0.0)
    sq = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + sig * sig / 2) * T) / sq
    return K * math.exp(-r * T) * ND.cdf(sq - d1) - S * ND.cdf(-d1)


def call_delta(S, K, sig, r, T):
    if sig <= 0 or S <= 0 or K <= 0 or T <= 1e-9:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + sig * sig / 2) * T) / (sig * math.sqrt(T))
    return ND.cdf(d1)


def k_for_d1(S, sig, r, T, d1):
    return S * math.exp((r + sig * sig / 2) * T - d1 * sig * math.sqrt(T))


def run(D, book, rf, cc_set, tgt_delta=0.42, cc_gate="always",
        skip_hot=False, put_entries=False):
    dates, px, dv, sma, sig_m, rfd, exp_map, ti, hi, hot = (
        D["dates"], D["px"], D["dv"], D["sma"], D["sig"], D["rf"],
        D["exp_map"], D["ti"], D["hi"], D["hot"])
    nD = len(dates)
    d1c = ND.inv_cdf(tgt_delta)
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
    oc = {}    # calls: t -> [K, exp_i, cov, prem_total]
    op = {}    # puts:  t -> [K, exp_i, cov, prem_total]
    c_gross = c_back = p_gross = 0.0
    cc_win = cc_loss = 0
    cc_loss_amt = 0.0
    put_n = put_assign = put_worthless = 0
    n_open = 0
    prem_pct = 0.0
    eq = np.empty(nD)

    def sell_call(t, p, sig, r, i):
        nonlocal pool, c_gross, n_open, prem_pct
        e = exp_map[i]
        T = max((dates[min(e, nD - 1)] - dates[i]).days, 1) / 365.0
        K = k_for_d1(p, sig, r, T, d1c)
        pr = bs_call(p, K, sig, r, T)
        pool += pr * sh[t]
        c_gross += pr * sh[t]
        n_open += 1
        prem_pct += pr / p * (30.0 / max(T * 365, 1))
        oc[t] = [K, e, sh[t], pr * sh[t]]

    def close_call(t, cost):
        nonlocal pool, c_back, cc_win, cc_loss, cc_loss_amt
        prem = oc[t][3]
        pool -= cost
        c_back += cost
        if cost > prem:
            cc_loss += 1
            cc_loss_amt += cost - prem
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

        # ---- manage open puts (entries) ----
        for t in list(op):
            K, e, cov, pr = op[t]
            p = px[i, ti[t]]
            if i >= e or np.isnan(p):
                if np.isnan(p):
                    del op[t]
                    continue
                if p < K:      # assigned: buy tranche at strike
                    sh[t] = sh.get(t, 0.0) + cov
                    cash -= K * cov
                    tr[t] = tr.get(t, 0) + 1
                    last_add[t] = d
                    put_assign += 1
                else:
                    put_worthless += 1
                del op[t]

        # ---- manage open calls ----
        for t in list(oc):
            K, e, cov, pr = oc[t]
            p = px[i, ti[t]]
            if np.isnan(p):
                if i >= e:
                    del oc[t]
                continue
            sig = sig_m[i, ti[t]]
            if i >= e:
                close_call(t, (p - K) * cov if p > K else 0.0)
                if t in cc_set and sig > 0:
                    ok = cc_gate == "always" or (
                        not np.isnan(sma[i, ti[t]]) and p > sma[i, ti[t]])
                    if cc_gate == "extnh":
                        ok = ok and not np.isnan(hi[i, ti[t]]) \
                            and p <= hi[i, ti[t]]
                    if skip_hot and hot[i, ti[t]]:
                        ok = False
                    if ok:
                        sell_call(t, p, sig, r, i)
                continue
            T = (dates[e if e < nD else nD - 1] - d).days / 365.0
            if call_delta(p, K, sig, r, T) > ROLL_DELTA:
                close_call(t, bs_call(p, K, sig, r, T) * cov)
                if sig > 0:
                    sell_call(t, p, sig, r, i)

        # ---- open new calls ----
        for t in cc_set:
            if t in oc or t not in ti or sh.get(t, 0) <= 0:
                continue
            p = px[i, ti[t]]
            sig = sig_m[i, ti[t]]
            if np.isnan(p) or np.isnan(sig) or sig <= 0:
                continue
            if cc_gate in ("ext", "extnh"):
                s = sma[i, ti[t]]
                if np.isnan(s) or p <= s:
                    continue
                if cc_gate == "extnh" and (np.isnan(hi[i, ti[t]])
                                           or p > hi[i, ti[t]]):
                    continue
            if skip_hot and hot[i, ti[t]]:
                continue
            sell_call(t, p, sig, r, i)

        # ---- tranche program (direct buy or via cash-secured put) ----
        for t in iv0:
            p = px[i, ti[t]]
            if np.isnan(p) or t in op:
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
            if not trigger or cash < TRANCHE:
                continue
            sig = sig_m[i, ti[t]]
            if put_entries and not np.isnan(sig) and sig > 0:
                e = exp_map[i]
                T = max((dates[min(e, nD - 1)] - d).days, 1) / 365.0
                K = k_for_d1(p, sig, r, T, D1_PUT)
                cov = TRANCHE / K
                pr = bs_put(p, K, sig, r, T)
                pool += pr * cov
                p_gross += pr * cov
                put_n += 1
                op[t] = [K, e, cov, pr * cov]
            else:
                sh[t] = sh.get(t, 0.0) + TRANCHE / p
                cash -= TRANCHE
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
        for t, (K, e, cov, _) in op.items():
            p = px[i, ti[t]]
            if np.isnan(p):
                continue
            T = (dates[e if e < nD else nD - 1] - d).days / 365.0
            mv -= bs_put(p, K, sig_m[i, ti[t]], r, T) * cov
        eq[i] = mv

    ser = pd.Series(eq, index=dates)
    tot_calls = cc_win + cc_loss
    meta = {"call_gross": round(c_gross), "call_back": round(c_back),
            "call_net": round(c_gross - c_back),
            "kept_pct": round(100 * (c_gross - c_back) / c_gross, 1)
            if c_gross else 0.0,
            "calls_total": tot_calls, "calls_profitable": cc_win,
            "calls_unprofitable": cc_loss,
            "unprofitable_pct": round(100 * cc_loss / tot_calls, 1)
            if tot_calls else 0.0,
            "loss_on_unprofitable": round(cc_loss_amt),
            "avg_prem_pct_30d": round(100 * prem_pct / n_open, 2)
            if n_open else 0.0,
            "put_gross": round(p_gross), "puts_sold": put_n,
            "puts_assigned": put_assign,
            "puts_worthless": put_worthless}
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
    hi_df = px_df.rolling(252).max().shift(1)
    mom = px_df.pct_change(126, fill_method=None)
    dgs10 = pd.read_csv(os.path.join(HERE, "dgs10.csv"),
                        index_col=0, parse_dates=True).iloc[:, 0]
    rf_daily = dgs10.reindex(px_df.index, method="ffill") / 100
    return px_df, dv_df, sma_df, sig_df, hi_df, mom, rf_daily


def arrays_for(year, start, px_df, dv_df, sma_df, sig_df, hi_df, mom,
               rf_daily, book):
    sub = px_df.loc[start:]
    dates = sub.index
    ti = {t: j for j, t in enumerate(px_df.columns)}
    # hottest-2 book stocks each day by trailing 26w return
    bm = mom[list(t for t in book if t in mom.columns)].loc[start:]
    rank = bm.rank(axis=1, ascending=False)
    hot_df = pd.DataFrame(False, index=dates, columns=px_df.columns)
    hot_df[bm.columns] = rank <= 2
    return {
        "dates": dates, "px": sub.values,
        "dv": dv_df.loc[start:].values,
        "sma": sma_df.loc[start:].values,
        "sig": sig_df.loc[start:].values,
        "rf": rf_daily.loc[start:].fillna(RF[year]).values,
        "ti": ti, "hi": hi_df.loc[start:].values,
        "hot": hot_df.values,
        "exp_map": np.searchsorted(
            dates.values, dates.values + np.timedelta64(DTE, "D")),
    }


def main():
    px_df, dv_df, sma_df, sig_df, hi_df, mom, rf_daily = prepare()
    out, curves = {}, {}
    for year, start in VINTAGES.items():
        book = load_book(year)
        arr = arrays_for(year, start, px_df, dv_df, sma_df, sig_df,
                         hi_df, mom, rf_daily, book)
        rfa = rf_daily.loc[start:].fillna(RF[year])
        allset = set(book)
        cfgs = [
            ("none", set(), .42, "always", False, False),
            ("cc_always", allset, .42, "always", False, False),
            ("cc_ext", allset, .42, "ext", False, False),
            ("cc_extnh", allset, .42, "extnh", False, False),
            ("cc_skiphot", allset, .42, "always", True, False),
            ("cc_extnh_skiphot", allset, .42, "extnh", True, False),
            ("put_entries_only", set(), .42, "always", False, True),
            ("wheel", allset, .42, "always", False, True),
            ("wheel_extnh", allset, .42, "extnh", False, True),
            ("wheel_extnh_skiphot", allset, .42, "extnh", True, True),
        ]
        vout = {}
        for name, cset, tdel, gate, shot, pent in cfgs:
            ser, meta = run(arr, book, RF[year], cset, tdel, gate,
                            shot, pent)
            c, dd, sp = stats_of(ser, rfa)
            vout[name] = {"cagr": c, "dd": dd, "sharpe": sp,
                          "final": round(ser.iloc[-1]), **meta}
            if name in ("none", "cc_always", "wheel"):
                curves[f"{year}_{name}"] = ser
        out[str(year)] = vout
        b = vout["none"]["cagr"]
        print(f"{year} base {b}%: " + " ".join(
            f"{n}:{v['cagr'] - b:+.2f}" for n, v in vout.items()
            if n != "none"), flush=True)

    json.dump(out, open(os.path.join(HERE, "cc_daily2_results.json"),
                        "w"), indent=1)
    pd.DataFrame(curves).to_csv(
        os.path.join(HERE, "cc_daily2_curves.csv"))


if __name__ == "__main__":
    main()
