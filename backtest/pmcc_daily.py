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
             tech_reentry=False, lev=1.0, initial=INITIAL):
    """lev: leverage multiple on notional. Each tranche of `budget` dollars
    buys call exposure on lev*budget/p shares (capped by what the budget
    can actually pay in premium). lev=1 -> share-equivalent (no leverage);
    higher lev -> more exposure per dollar; full_tranche -> maximum (the
    entire budget is spent on premium). Short calls scale automatically
    because they are written on the full call coverage.
    tech_reentry: at 30 DTE the long is SOLD (not rolled immediately);
    the freed money waits in cash and a new long call is only bought when
    price is back above the 200-day SMA (uptrend, per the DS PDF's trend
    filter). No hindsight -- the SMA is known each day."""
    if full_tranche:
        lev = float("inf")
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

    lc = {}                    # t -> list of lots [K, exp_i, cov, cost]
    backing = {}               # t -> reserved cash behind the notional
    waiting = {}               # tech_reentry: t -> reserved $ awaiting uptrend
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
        return sum(l[2] for l in lc.get(t, []))

    def free_cash():
        return cash - sum(backing.values()) - sum(waiting.values())

    def open_long(t, p, sig, r, i, budget):
        """Buy delta-LONG_D calls with `budget` dollars of NOTIONAL.
        Non-full mode: cov = budget/p shares of exposure; the unspent part
        of the budget is locked as backing (no re-use = no leverage).
        full_tranche: the whole budget is spent on premium (levered)."""
        nonlocal cash, long_paid, n_longroll
        avail = free_cash()
        budget = min(budget, avail)
        if budget <= 0:
            return False
        e = exp_i(i, long_dte)
        T = max((dates[min(e, nD - 1)] - dates[i]).days, 1) / 365.0
        K = k_for_d1(p, sig, r, T, d1_long)
        prc = bs_call(p, K, sig, r, T)
        if prc <= 0:
            return False
        cov_max = budget / prc             # most exposure the budget can buy
        cov = cov_max if math.isinf(lev) else min(lev * budget / p, cov_max)
        cost = prc * cov
        cash -= cost
        long_paid += cost
        # lock the unspent remainder of the tranche so it cannot be
        # double-spent elsewhere (the tranche's capital stays committed)
        backing[t] = backing.get(t, 0.0) + max(budget - cost, 0.0)
        lc.setdefault(t, []).append([K, e, cov, cost])
        return True

    def close_lot(t, j, i, r):
        """Sell lot j of t at model value; release its share of backing."""
        nonlocal cash, long_recv
        K, e, cov, cost = lc[t][j]
        p = px[i, ti[t]]
        T = max((dates[min(e, nD - 1)] - dates[i]).days, 0) / 365.0
        val = bs_call(p, K, sig_m[i, ti[t]], r, T) * cov
        cash += val
        long_recv += val
        if t in backing:
            tot = sum(l[2] for l in lc[t])
            rel = backing[t] * (cov / tot) if tot > 0 else backing[t]
            backing[t] -= rel
        lc[t].pop(j)
        if not lc[t]:
            del lc[t]
            backing.pop(t, None)
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

        # ---- manage long calls (lot by lot) ----
        for t in list(lc):
            p = px[i, ti[t]]
            if np.isnan(p):
                continue
            sig = sig_m[i, ti[t]]
            # hammer cut-loss: whole-position value at 50% of total cost
            if style == "hammer" and not np.isnan(sig) and sig > 0:
                tot_val = tot_cost = 0.0
                for (K, e, cov, cost) in lc[t]:
                    T = max((dates[min(e, nD - 1)] - d).days, 0) / 365.0
                    tot_val += bs_call(p, K, sig, r, T) * cov
                    tot_cost += cost
                sc = 0.0
                if t in oc:
                    Ks, es, cs, prs = oc[t]
                    Ts = max((dates[min(es, nD - 1)] - d).days, 0) / 365.0
                    sc = bs_call(p, Ks, sig, r, Ts) * cs
                if tot_val - sc <= 0.5 * tot_cost - 1e-9:
                    if t in oc:
                        close_short(t, sc)
                    while t in lc:
                        close_lot(t, 0, i, r)
                    stopped[t] = True
                    n_cutloss += 1
                    continue
            # roll / convert lots nearing expiry
            # Snapshot lot count: lots re-opened TODAY by open_long are
            # appended at the end and must NOT be reprocessed in the same
            # pass (prevents endless same-day roll churn).
            j = 0
            n_lots_now = len(lc.get(t, []))
            while t in lc and j < min(n_lots_now, len(lc[t])):
                K, e, cov, cost = lc[t][j]
                days_left = (dates[min(e, nD - 1)] - d).days
                # e >= nD: true expiry lies beyond the data window -- nothing
                # to roll (prevents endless same-day re-rolls in the final
                # weeks of the sample)
                if days_left > ROLL_LONG_AT or e >= nD:
                    j += 1
                    continue
                if convert:
                    val, cov_, _ = close_lot(t, j, i, r)
                    take = min(cov_, cash / p if p > 0 else 0)
                    sh[t] = sh.get(t, 0.0) + take
                    cash -= take * p
                    n_convert += 1
                elif tech_reentry:
                    # sell the lot; park its notional until uptrend returns
                    val, cov_, _ = close_lot(t, j, i, r)
                    park = min(cov_ * p, max(cash - sum(backing.values())
                                             - sum(waiting.values()), 0.0))
                    waiting[t] = waiting.get(t, 0.0) + park
                else:
                    if np.isnan(sig) or sig <= 0:
                        j += 1
                        continue
                    val, cov_, _ = close_lot(t, j, i, r)
                    # re-establish the SAME notional exposure at this lev
                    if math.isinf(lev):
                        budget = val
                    else:
                        budget = cov_ * p / lev
                    open_long(t, p, sig, r, i, budget)
                    n_longroll += 1
            # trim short coverage if longs shrank
            if t in oc:
                oc[t][2] = min(oc[t][2], cov_of(t) + sh.get(t, 0.0))

        # ---- tech re-entry: buy the long back when p > 200d SMA ----
        if tech_reentry:
            for t in list(waiting):
                p = px[i, ti[t]]
                s = sma[i, ti[t]]
                sig = sig_m[i, ti[t]]
                if np.isnan(p) or np.isnan(s) or np.isnan(sig) or sig <= 0:
                    continue
                if p > s:
                    budget = waiting.pop(t)
                    if open_long(t, p, sig, r, i, budget):
                        n_longroll += 1
                    else:
                        waiting[t] = budget

        # ---- hammer re-entry after cut-loss ----
        for t in list(stopped):
            p = px[i, ti[t]]
            s = sma[i, ti[t]]
            sig = sig_m[i, ti[t]]
            if np.isnan(p) or np.isnan(s) or np.isnan(sig) or sig <= 0:
                continue
            if p > s:                       # neutral-to-bullish again
                if open_long(t, p, sig, r, i, tranche):
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
            if not trigger or free_cash() < tranche:
                continue
            sig = sig_m[i, ti[t]]
            if np.isnan(sig) or sig <= 0:
                continue
            if open_long(t, p, sig, r, i, tranche):
                tr[t] = ntr + 1
                last_add[t] = d

        # ---- reinvest pool: add long-call coverage on cheapest vs IV ----
        if pool > 0:
            cash += pool          # premiums land in cash first
            pool = 0.0
        # deploy excess free cash beyond a reserve into extra coverage
        reserve = tranche * 2     # keep dry powder for pending tranches
        fc = free_cash()
        if fc > reserve * 1.5:
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
                    open_long(bt, p, sig, r, i, fc - reserve)

        # ---- mark ----
        mv = cash + pool
        for t, s_ in sh.items():
            if s_ > 0:
                p = px[i, ti[t]]
                if not np.isnan(p):
                    mv += s_ * p
        for t, lots in lc.items():
            p = px[i, ti[t]]
            if np.isnan(p):
                continue
            for (K, e, cov, _) in lots:
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
            for name, style, conv, ft, tech, lev in [
                    ("ds1", "ds1", False, False, False, 1.0),
                    ("ds1_convert", "ds1", True, False, False, 1.0),
                    ("ds1_lev2", "ds1", False, False, False, 2.0),
                    ("ds1_lev3", "ds1", False, False, False, 3.0),
                    ("ds1_lev2_conv", "ds1", True, False, False, 2.0),
                    ("ds1_lev3_conv", "ds1", True, False, False, 3.0),
                    ("hammer", "hammer", False, False, False, 1.0),
                    ("hammer_lev2", "hammer", False, False, False, 2.0),
                    ("ds1_full", "ds1", False, True, False, 1.0)]:
                ser, meta = run_pmcc(arr, book, RF[year], style, conv, ft,
                                     tech_reentry=tech, lev=lev)
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
