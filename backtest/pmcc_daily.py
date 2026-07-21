"""PMCC study: replace share ownership with a long call, keep selling calls.

User request: for the stocks that normally qualify for covered calls (in our
finding: the whole book), test replacing the 100-shares leg with a long call
per the two uploaded PDFs, on the 1990/2000/2015 books at $100k start.

Configs (all day-by-day, real closes/dividend dates, calibrated BS):
  stock_cc     reference: own shares + delta-42 35-DTE calls (cc_daily3 'all')
  ds1          DS 1.0 / classic PMCC (Diagonal Spread 2024.pdf):
                 long  = delta-80 call, ~150 DTE ("3-6 months")
                 short = delta-42 call, 35 DTE, roll at delta 0.80
                 long rolled when < 30 DTE remain (sell old, buy new d80)
                 SAME notional share exposure as the stock version; the
                 ~80% of the tranche not spent on the call sits in cash
                 earning the T-bill-ish rate; dividends are forfeited.
  ds1_convert  same, but when the long call is down to < 30 DTE we CONVERT:
                 sell the call and buy the same number of real shares with
                 the reserved cash (user's idea) -> from then on that stock
                 is a plain covered-call position collecting dividends.
  hammer       DS Hammer (DS Hammer.pdf):
                 long  = delta-55 call, ~180 DTE ("ATM, 3-9 months")
                 short = delta-22 call, 14 DTE ("1-SD, 1-3 weeks"),
                         roll at expiry or delta > 0.80
                 cut-loss: close the pair at 50% of the debit, sit in cash,
                 re-enter when price > 200d SMA (per 'neutral-to-bullish
                 setup' rule); long rolled when < 30 DTE remain.
  ds1_full     all-in variant: the ENTIRE tranche is spent on delta-80
                 calls (~5x notional exposure, max loss = the tranche).
                 This is the 'fraction of capital -> magnified ROI' pitch
                 taken literally.

Everything else (IV-based tranche buying, 56-day spacing, premium pool
reinvested into the cheapest below-IV stock, $100k start) matches
cc_daily3.py so the comparison is apples-to-apples.
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
from cc_daily3 import (INITIAL, N16, GAP_DAYS, VINTAGES, RF, ND,  # noqa: E402
                       bs_call, call_delta, k_for_d1, load_books3,
                       prepare, arrays_for, stats_of, run as run_cc)

LONG_DTE = {"ds1": 150, "hammer": 180}
LONG_D = {"ds1": 0.80, "hammer": 0.55}
SHORT_DTE = {"ds1": 35, "hammer": 14}
SHORT_D = {"ds1": 0.42, "hammer": 0.22}
ROLL_LONG_AT = 30          # days remaining on the long call
ROLL_SHORT_DELTA = 0.80


def run_pmcc(D, book, rf, style="ds1", convert=False, full_tranche=False,
             initial=INITIAL):
    dates, px, dv, sma, sig_m, rfd, ti = (
        D["dates"], D["px"], D["dv"], D["sma"], D["sig"], D["rf"], D["ti"])
    nD = len(dates)
    dvals = dates.values
    cap = initial / N16
    tranche = cap / 3.0
    d1_long = ND.inv_cdf(LONG_D[style])
    d1_short = ND.inv_cdf(SHORT_D[style])
    long_dte, short_dte = LONG_DTE[style], SHORT_DTE[style]

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

    def exp_i(i, days):
        return int(np.searchsorted(dvals, dvals[i] + np.timedelta64(days, "D")))

    lc = {}                    # t -> [K, exp_i, cov, cost]      long calls
    sh = {t: 0.0 for t in iv0}  # real shares (after conversion)
    oc = {}                    # t -> [K, exp_i, cov, prem]      short calls
    stopped = {}               # hammer cut-loss parking: t -> True
    tr, last_add = {}, {}
    cash = initial
    pool = 0.0
    c_gross = c_back = 0.0
    cc_win = cc_loss = 0
    long_paid = long_recv = 0.0
    div_total = interest = 0.0
    n_convert = n_cutloss = n_longroll = 0
    eq = np.empty(nD)

    def cov_of(t):
        return lc[t][2] if t in lc else 0.0

    def open_long(t, p, sig, r, i, budget=None):
        """Buy delta-LONG_D call; cov = notional shares (tranche/p) unless
        full_tranche (spend the whole budget on calls)."""
        nonlocal cash, long_paid, n_longroll
        e = exp_i(i, long_dte)
        T = max((dates[min(e, nD - 1)] - dates[i]).days, 1) / 365.0
        K = k_for_d1(p, sig, r, T, d1_long)
        prc = bs_call(p, K, sig, r, T)
        if prc <= 0:
            return False
        b = tranche if budget is None else budget
        cov = (b / prc) if full_tranche else (b / p)
        cost = prc * cov
        if cost > cash + 1e-9:
            cov = cash / prc if full_tranche else cash / p
            cost = prc * cov
            if cov <= 0:
                return False
        cash -= cost
        long_paid += cost
        if t in lc:                        # rolling: merge
            lc[t] = [K, e, lc[t][2] + cov, lc[t][3] + cost]
        else:
            lc[t] = [K, e, cov, cost]
        return True

    def close_long(t, i, r):
        nonlocal cash, long_recv
        K, e, cov, cost = lc[t]
        p = px[i, ti[t]]
        T = max((dates[min(e, nD - 1)] - dates[i]).days, 0) / 365.0
        val = bs_call(p, K, sig_m[i, ti[t]], r, T) * cov
        cash += val
        long_recv += val
        del lc[t]
        return val, cov, cost

    def sell_short(t, p, sig, r, i):
        nonlocal pool, c_gross
        c = cov_of(t) + sh.get(t, 0.0)
        if c <= 0:
            return
        e = exp_i(i, short_dte)
        T = max((dates[min(e, nD - 1)] - dates[i]).days, 1) / 365.0
        K = k_for_d1(p, sig, r, T, d1_short)
        pr = bs_call(p, K, sig, r, T)
        pool += pr * c
        c_gross += pr * c
        oc[t] = [K, e, c, pr * c]

    def close_short(t, cost):
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
        # cash interest (daily, T-bill-ish = DGS10 path, consistent w/ rest)
        if cash > 0:
            inc = cash * r / 252
            cash += inc
            interest += inc
        # dividends only on real shares
        for t in sh:
            if sh[t] > 0:
                x = dv[i, ti[t]]
                if x > 0:
                    pool += sh[t] * x
                    div_total += sh[t] * x

        # ---- manage short calls ----
        for t in list(oc):
            K, e, c, pr = oc[t]
            p = px[i, ti[t]]
            if np.isnan(p):
                if i >= e:
                    del oc[t]
                continue
            sig = sig_m[i, ti[t]]
            if i >= e:
                close_short(t, (p - K) * c if p > K else 0.0)
                if sig > 0 and (cov_of(t) + sh.get(t, 0)) > 0:
                    sell_short(t, p, sig, r, i)
                continue
            T = (dates[e if e < nD else nD - 1] - d).days / 365.0
            if call_delta(p, K, sig, r, T) > ROLL_SHORT_DELTA:
                close_short(t, bs_call(p, K, sig, r, T) * c)
                if sig > 0 and (cov_of(t) + sh.get(t, 0)) > 0:
                    sell_short(t, p, sig, r, i)

        # ---- manage long calls ----
        for t in list(lc):
            p = px[i, ti[t]]
            if np.isnan(p):
                continue
            sig = sig_m[i, ti[t]]
            K, e, cov, cost = lc[t]
            days_left = (dates[min(e, nD - 1)] - d).days
            # hammer cut-loss: pair value dropped to 50% of what we paid
            if style == "hammer" and not np.isnan(sig) and sig > 0:
                T = max(days_left, 0) / 365.0
                val = bs_call(p, K, sig, r, T) * cov
                sc = 0.0
                if t in oc:
                    Ks, es, cs, prs = oc[t]
                    Ts = max((dates[min(es, nD - 1)] - d).days, 0) / 365.0
                    sc = bs_call(p, Ks, sig, r, Ts) * cs
                if val - sc <= 0.5 * cost - 1e-9:
                    if t in oc:
                        close_short(t, sc)
                    close_long(t, i, r)
                    stopped[t] = True
                    n_cutloss += 1
                    continue
            if days_left <= ROLL_LONG_AT:
                if convert:
                    val, cov_, _ = close_long(t, i, r)
                    take = min(cov_, cash / p if p > 0 else 0)
                    sh[t] = sh.get(t, 0.0) + take
                    cash -= take * p
                    n_convert += 1
                    if t in oc and take < cov_ * 0.999:   # trim coverage
                        oc[t][2] = min(oc[t][2], take + cov_of(t))
                else:
                    if np.isnan(sig) or sig <= 0:
                        continue
                    close_long(t, i, r)
                    open_long(t, p, sig, r, i,
                              budget=tranche if full_tranche else None)
                    n_longroll += 1

        # ---- hammer re-entry after cut-loss ----
        for t in list(stopped):
            p = px[i, ti[t]]
            s = sma[i, ti[t]]
            sig = sig_m[i, ti[t]]
            if np.isnan(p) or np.isnan(s) or np.isnan(sig) or sig <= 0:
                continue
            if p > s:                       # neutral-to-bullish again
                if open_long(t, p, sig, r, i):
                    del stopped[t]

        # ---- open short calls where uncovered ----
        for t in iv0:
            if t in oc:
                continue
            c = cov_of(t) + sh.get(t, 0.0)
            if c <= 0:
                continue
            p = px[i, ti[t]]
            sig = sig_m[i, ti[t]]
            if np.isnan(p) or np.isnan(sig) or sig <= 0:
                continue
            sell_short(t, p, sig, r, i)

        # ---- tranche program: entries open LONG CALLS not shares ----
        for t in iv0:
            p = px[i, ti[t]]
            if np.isnan(p) or t in stopped:
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
            if not trigger or cash < tranche * (1.0 if full_tranche else 0.35):
                continue
            sig = sig_m[i, ti[t]]
            if np.isnan(sig) or sig <= 0:
                continue
            if open_long(t, p, sig, r, i):
                tr[t] = ntr + 1
                last_add[t] = d

        # ---- reinvest pool: add long-call coverage on cheapest vs IV ----
        if pool > 0:
            cash += pool          # premiums land in cash first
            pool = 0.0
        # deploy excess cash beyond the tranche reserve into extra coverage
        reserve = tranche * 2     # keep dry powder for pending tranches
        if cash > reserve * 1.5:
            best, bt = 9e9, None
            for t in iv0:
                if t in stopped:
                    continue
                p = px[i, ti[t]]
                s = sma[i, ti[t]]
                if np.isnan(p) or np.isnan(s):
                    continue
                ivd = iv_at(t, i)
                if p <= s * 1.01 and p < ivd and p / ivd < best:
                    best, bt = p / ivd, t
            if bt is not None:
                p = px[i, ti[bt]]
                sig = sig_m[i, ti[bt]]
                if not np.isnan(sig) and sig > 0:
                    spend = cash - reserve
                    open_long(bt, p, sig, r, i,
                              budget=spend if full_tranche else spend)

        # ---- mark ----
        mv = cash + pool
        for t, s_ in sh.items():
            if s_ > 0:
                p = px[i, ti[t]]
                if not np.isnan(p):
                    mv += s_ * p
        for t, (K, e, cov, _) in lc.items():
            p = px[i, ti[t]]
            if np.isnan(p):
                continue
            T = max((dates[min(e, nD - 1)] - d).days, 0) / 365.0
            mv += bs_call(p, K, sig_m[i, ti[t]], r, T) * cov
        for t, (K, e, c, _) in oc.items():
            p = px[i, ti[t]]
            if np.isnan(p):
                continue
            T = max((dates[min(e, nD - 1)] - d).days, 0) / 365.0
            mv -= bs_call(p, K, sig_m[i, ti[t]], r, T) * c
        eq[i] = mv

    ser = pd.Series(eq, index=dates)
    tot = cc_win + cc_loss
    meta = {"short_prem_in": round(c_gross), "short_paid_out": round(c_back),
            "short_net": round(c_gross - c_back),
            "shorts_total": tot, "shorts_lost": cc_loss,
            "long_paid": round(long_paid), "long_recv": round(long_recv),
            "long_net_decay": round(long_paid - long_recv),
            "dividends": round(div_total), "interest": round(interest),
            "long_rolls": n_longroll, "converts": n_convert,
            "cutlosses": n_cutloss}
    return ser, meta


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
            vout = {}
            # reference: stock + CC (identical to cc_daily3 'all')
            ser, meta = run_cc(arr, book, RF[year], allset)
            c, dd, sp = stats_of(ser, rfa)
            vout["stock_cc"] = {"cagr": c, "dd": dd, "sharpe": sp,
                                "final": round(ser.iloc[-1])}
            for name, style, conv, ft in [
                    ("ds1", "ds1", False, False),
                    ("ds1_convert", "ds1", True, False),
                    ("hammer", "hammer", False, False),
                    ("ds1_full", "ds1", False, True)]:
                ser, meta = run_pmcc(arr, book, RF[year], style, conv, ft)
                c, dd, sp = stats_of(ser, rfa)
                vout[name] = {"cagr": c, "dd": dd, "sharpe": sp,
                              "final": round(ser.iloc[-1]), **meta}
                if name in ("ds1", "hammer"):
                    curves[f"{year}_{bname}_{name}"] = ser
            out[f"{year}_{bname}"] = vout
            print(f"{year} {bname:6}: " + " ".join(
                f"{n}:{v['cagr']:.2f}%/dd{v['dd']:.0f}"
                for n, v in vout.items()), flush=True)

    json.dump(out, open(os.path.join(HERE, "pmcc_results.json"), "w"),
              indent=1)
    pd.DataFrame(curves).to_csv(os.path.join(HERE, "pmcc_curves.csv"))


if __name__ == "__main__":
    main()
