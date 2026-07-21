"""Day-by-day covered-call simulation on real daily prices.

User asks: calculate what actually happens day by day; keep as much of
the collected credit as possible (reduce what we pay to close losing
calls); test Adam's 'prefer Dow stocks'; keep it real.

Mechanics per calendar day (real trading days from daily_unadj.csv):
  * dividends credited on actual ex-dates
  * open a 35-calendar-day call at target delta when uncovered
  * exit variants:
      roll80 : buy back when delta > 0.80 or ITM with <=5 days left,
               sell new call immediately (Adam's PDF scenarios 2+3)
      hold   : NEVER buy back early. At expiry, if ITM the shares are
               called away at K and immediately repurchased at market
               (cost = intrinsic only -- you never pay time value to
               close). Then sell a new call. (Adam's scenario 4.)
  * strike deltas tested: 0.42 (Adam) and 0.25 (further out: fewer
    calls finish ITM -> keep more of the credit)
  * sigma = 130-trading-day realized vol * daily VIX/SPX-RV ratio
    (same data-derived IV calibration as before, now daily)
  * haircut h: sell premium at (1-h)*BS, buy back at (1+h)*BS
  * per-call ledger: every closed call logged win/loss so we can say
    exactly how many calls made money and how many gave credit back.

Tranche program: same VMI rules on the daily grid (200d SMA ~ 40w,
min 40 trading days between adds). Pool (divs + CC income) reinvested
with discipline: on-support & under-IV, cheapest vs IV.
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
GAP_D = 40                 # trading days (~8 weeks)
SMA_D = 200                # ~40 weeks
VOL_D = 130                # ~26 weeks
END = "2026-07-18"
ND = NormalDist()
DTE = 35                   # calendar days
ROLL_DELTA = 0.80

VINTAGES = {
    1990: "1990-01-05", 1995: "1995-01-06", 2000: "2000-01-07",
    2005: "2005-01-07", 2010: "2010-01-08", 2015: "2015-01-09",
    2020: "2020-01-10",
}
RF = {1990: .0794, 1995: .0788, 2000: .065, 2005: .0423,
      2010: .0385, 2015: .0212, 2020: .0188}

# Era-correct Dow Jones 30 members within each vintage book (public
# historical membership record, no hindsight):
DOW = {
    1990: {"KO", "MCD", "AXP"},                       # DIS added 1991
    1995: {"KO", "MCD", "AXP", "DIS"},
    2000: set(),                                       # none in that book
    2005: {"AXP", "HD", "JNJ", "MO", "MSFT", "WMT"},
    2010: {"MSFT", "MCD", "WMT"},                      # AAPL added 2015
    2015: set(),                                       # AAPL added Mar-15
    2020: {"UNH"},
}


def load_book(year):
    if year == 2000:
        return {t: v for t, v in B2000_GRO.items()}
    d = json.load(open(os.path.join(HERE, f"books_growth_{year}.json")))
    return {t.replace(".", "-"): (v["pe"], v["g"], v["beta"], v["ocf_mult"])
            for t, v in d["growth_book"].items()}


def bs_call(S, K, sig, r, T):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    st = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r + sig * sig / 2) * T) / st
    return S * ND.cdf(d1) - K * math.exp(-r * T) * ND.cdf(d1 - st)


def call_delta(S, K, sig, r, T):
    if T <= 0 or sig <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + sig * sig / 2) * T) / (sig * math.sqrt(T))
    return ND.cdf(d1)


def run(D, book, rf, cc_set, exit_mode, tgt_delta, haircut, label):
    """D: dict of per-ticker numpy dicts + shared arrays."""
    dates = D["dates"]
    nd = len(dates)
    days_arr = D["days_frac"]          # cumulative calendar days
    rfr = D["rfr"]
    d1t = ND.inv_cdf(tgt_delta)

    iv0, g_iv = {}, {}
    for t in book:
        if t not in D["px"]:
            continue
        p0 = D["first_px"][t]
        if p0 is None:
            continue
        pe, g, b, m = book[t]
        iv0[t] = p0 * dcf_factor(g, b, rf) * m / pe
        g_iv[t] = min(g, 0.10)

    sh, tr, last = {}, {}, {}
    cash, pool = INITIAL, 0.0
    open_cc = {}     # t -> [K, exp_day(calendar), shares, prem_collected]
    gross, paid_close = 0.0, 0.0
    n_win, n_loss, win_amt, loss_amt = 0, 0, 0.0, 0.0
    prem_pct_sum, n_opens = 0.0, 0
    eq = np.empty(nd)

    px, sma, sig_a, dv = D["px"], D["sma"], D["sig"], D["dv"]

    def iv_at(t, i):
        return iv0[t] * (1 + g_iv[t]) ** (days_arr[i] / 365.25)

    def sell_new(t, p, sg, r_now, i):
        nonlocal pool, gross, prem_pct_sum, n_opens
        K = p * math.exp((r_now + sg * sg / 2) * (DTE / 365.0)
                         - d1t * sg * math.sqrt(DTE / 365.0))
        prem_ps = bs_call(p, K, sg, r_now, DTE / 365.0) * (1 - haircut)
        cred = prem_ps * sh[t]
        pool += cred
        gross += cred
        prem_pct_sum += (prem_ps / p) * (30.0 / DTE)
        n_opens += 1
        open_cc[t] = [K, days_arr[i] + DTE, sh[t], cred]

    for i in range(nd):
        today_cal = days_arr[i]
        r_now = rfr[i]
        # dividends (actual ex-dates)
        for t, s_ in sh.items():
            if t in dv:
                x = dv[t][i]
                if x > 0:
                    pool += s_ * x

        # --- manage open calls ---
        for t in list(open_cc):
            K, exp_cal, cov, cred = open_cc[t]
            p = px[t][i]
            if np.isnan(p):
                if today_cal >= exp_cal:
                    del open_cc[t]
                continue
            T_rem = max(exp_cal - today_cal, 0.0) / 365.0
            sg = sig_a[t][i]
            expired = today_cal >= exp_cal
            close_cost = None
            if expired:
                close_cost = max(p - K, 0.0) * cov     # intrinsic only
            elif exit_mode == "roll80":
                dlt = call_delta(p, K, sg, r_now, T_rem) \
                    if not np.isnan(sg) else (1.0 if p > K else 0.0)
                if dlt > ROLL_DELTA or (exp_cal - today_cal <= 5 and p > K):
                    close_cost = bs_call(p, K, sg if not np.isnan(sg)
                                         else 0.0, r_now, T_rem) \
                        * cov * (1 + haircut)
            if close_cost is None:
                continue
            pool -= close_cost
            paid_close += close_cost
            pnl = cred - close_cost
            if pnl >= 0:
                n_win += 1; win_amt += pnl
            else:
                n_loss += 1; loss_amt -= pnl
            del open_cc[t]
            if t in cc_set and not np.isnan(sg) and sg > 0:
                sell_new(t, p, sg, r_now, i)

        # fresh opens (always-covered; rolls above already re-sell)
        for t in cc_set:
            if t in open_cc or t not in sh or t not in px:
                continue
            p, sg = px[t][i], sig_a[t][i]
            if np.isnan(p) or np.isnan(sg) or sg <= 0:
                continue
            sell_new(t, p, sg, r_now, i)

        # tranche program
        for t in iv0:
            p = px[t][i]
            if np.isnan(p):
                continue
            if t not in tr:
                if p < iv_at(t, i) and cash >= TRANCHE:
                    sh[t] = sh.get(t, 0) + TRANCHE / p
                    tr[t] = 1; last[t] = i; cash -= TRANCHE
            elif tr[t] < 3 and i - last[t] >= GAP_D:
                s = sma[t][i]
                if not np.isnan(s) and p <= s * 1.01 \
                        and p < iv_at(t, i) and cash >= TRANCHE:
                    sh[t] += TRANCHE / p; tr[t] += 1; last[t] = i
                    cash -= TRANCHE

        # disciplined reinvest of pool
        if pool > 0:
            best, bt, bp = None, None, None
            for t in iv0:
                p, s = px[t][i], sma[t][i]
                if np.isnan(p) or np.isnan(s):
                    continue
                ivd = iv_at(t, i)
                if p <= s * 1.01 and p < ivd:
                    ratio = p / ivd
                    if best is None or ratio < best:
                        best, bt, bp = ratio, t, p
            if bt is not None:
                sh[bt] = sh.get(bt, 0) + pool / bp
                pool = 0.0

        # mark-to-market
        mv = cash + pool
        for t, s_ in sh.items():
            p = px[t][i]
            if np.isnan(p):
                continue
            mv += s_ * p
            if t in open_cc:
                K, exp_cal, cov, _ = open_cc[t]
                sg = sig_a[t][i]
                T_rem = max(exp_cal - today_cal, 0.0) / 365.0
                mv -= bs_call(p, K, sg if not np.isnan(sg) else 0.0,
                              r_now, T_rem) * cov
        eq[i] = mv

    ser = pd.Series(eq, index=dates, name=label)
    yrs = (dates[-1] - dates[0]).days / 365.25
    meta = {
        "cagr_pct": round(((ser.iloc[-1] / INITIAL) ** (1 / yrs) - 1) * 100,
                          2),
        "final": round(float(ser.iloc[-1])),
        "maxdd_pct": round(float((ser / ser.cummax() - 1).min() * 100), 1),
        "gross_credit": round(gross),
        "paid_to_close": round(paid_close),
        "net_kept": round(gross - paid_close),
        "kept_pct_of_gross": round(100 * (gross - paid_close) / gross, 1)
        if gross else 0.0,
        "calls_won": n_win, "calls_lost": n_loss,
        "won_amt": round(win_amt), "lost_amt": round(loss_amt),
        "avg_prem_pct_30d": round(100 * prem_pct_sum / n_opens, 2)
        if n_opens else 0.0,
    }
    return ser, meta


def prep(year, start, px_all, dv_all, sma_all, sig_all, rfr_all):
    dates = px_all.loc[start:END].index
    idx = px_all.index.get_indexer(dates)
    D = {"dates": dates,
         "days_frac": np.array([(d - dates[0]).days for d in dates],
                               dtype=float),
         "rfr": rfr_all.reindex(dates).ffill().fillna(RF[year]).values,
         "px": {}, "sma": {}, "sig": {}, "dv": {}, "first_px": {}}
    for t in px_all.columns:
        D["px"][t] = px_all[t].values[idx]
        D["sma"][t] = sma_all[t].values[idx]
        D["sig"][t] = sig_all[t].values[idx]
        s = px_all[t].loc[start:].dropna()
        D["first_px"][t] = float(s.iloc[0]) if not s.empty else None
        if t in dv_all.columns:
            D["dv"][t] = dv_all[t].reindex(dates).fillna(0.0).values
    return D


def main():
    px = pd.read_csv(os.path.join(HERE, "daily_unadj.csv"),
                     index_col=0, parse_dates=True).loc[:END]
    dv = pd.read_csv(os.path.join(HERE, "daily_divs.csv"),
                     index_col=0, parse_dates=True).loc[:END]
    sma = px.rolling(SMA_D).mean()
    ret = px.pct_change(fill_method=None)
    rv = ret.rolling(VOL_D).std() * math.sqrt(252)
    # daily IV calibration: VIX / SPX realized vol, clipped, ffilled
    upl = (px["^VIX"] / 100.0 / rv["^GSPC"]).clip(0.8, 2.0).ffill()
    upl = upl.fillna(1.0)
    sig = rv.mul(upl, axis=0)
    dgs10 = pd.read_csv(os.path.join(HERE, "dgs10.csv"),
                        index_col=0, parse_dates=True).iloc[:, 0] / 100.0

    base = json.load(open(os.path.join(HERE, "div_results.json")))
    out = {}
    curves = {}
    for year, start in VINTAGES.items():
        book = load_book(year)
        D = prep(year, start, px, dv, sma, sig, dgs10)
        gs = {t: v[1] for t, v in book.items()}
        betas = {t: v[2] for t, v in book.items()}
        med_b = sorted(betas.values())[len(betas) // 2]
        g25lb = {t for t in book if gs[t] <= .25 and betas[t] <= med_b}
        dow = DOW[year] & set(book)
        grid = [
            ("ideal|roll80|d42", g25lb, "roll80", 0.42, 0.0),
            ("ideal|hold|d42",   g25lb, "hold",   0.42, 0.0),
            ("ideal|hold|d25",   g25lb, "hold",   0.25, 0.0),
            ("ideal|roll80|d25", g25lb, "roll80", 0.25, 0.0),
            ("all|roll80|d42",   set(book), "roll80", 0.42, 0.0),
            ("all|hold|d42",     set(book), "hold",   0.42, 0.0),
            ("all|hold|d25",     set(book), "hold",   0.25, 0.0),
            ("ideal|hold|d42|h5", g25lb, "hold",  0.42, 0.05),
            ("ideal|hold|d25|h5", g25lb, "hold",  0.25, 0.05),
            ("all|hold|d25|h5",  set(book), "hold", 0.25, 0.05),
        ]
        if dow:
            grid.append((f"dow|hold|d42", dow, "hold", 0.42, 0.0))
        vout = {"none": {
            "cagr_pct": base[str(year)]["growth"]["cagr_pct"],
            "final": base[str(year)]["growth"]["final"]}}
        for name, cset, mode, tdel, hc in grid:
            ser, meta = run(D, book, RF[year], cset, mode, tdel, hc,
                            f"{year} {name}")
            meta["n_cc_stocks"] = len(cset)
            vout[name] = meta
            if name == "ideal|hold|d25|h5":
                curves[str(year)] = ser
        out[str(year)] = vout
        b = vout["none"]["cagr_pct"]
        print(f"{year} (base {b:.2f}%): " + " ".join(
            f"{n}:{v['cagr_pct'] - b:+.2f}" for n, v in vout.items()
            if n != "none"), flush=True)

    json.dump(out, open(os.path.join(HERE, "cc_daily_results.json"), "w"),
              indent=1)
    pd.DataFrame(curves).to_csv(os.path.join(HERE, "cc_daily_curves.csv"))


if __name__ == "__main__":
    main()
