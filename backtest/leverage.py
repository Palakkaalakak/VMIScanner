"""UPRO-style daily-rebalanced leverage study on the 2000-2013 VMI accounts.

Method (matches how 2x/3x ETFs actually work):
  1. Reconstruct DAILY portfolio equity by replaying the exact trade log
     (trades.json) on daily adjusted closes -> daily unlevered returns r_t.
  2. Levered equity: E_t = E_{t-1} * (1 + Y*r_t - (Y-1)*rf_t - fee_t)
       Y      = leverage factor, rebalanced daily (constant exposure)
       rf_t   = daily financing cost on the borrowed (Y-1): 3-month T-bill
                (era-accurate monthly series, ~6% in 2000 -> ~0% in 2013)
       fee_t  = 0.95%/yr fund expense ratio (UPRO-like)
     Equity floored at 0 — max loss 100%, like a levered ETF.
  3. DISQUALIFIED if equity ever drops 99%+ from peak (economically blown;
     also flags true wipeouts).
Tested Y = 1.0 (check), 1.5 ... 5.0 step 0.5, on Defensive, Growth, SPY.
Outputs: leverage_results.json, daily equity CSVs for charting.
"""
import json
import numpy as np
import pandas as pd
import yfinance as yf

START, END = "2000-01-01", "2013-12-31"
INITIAL = 1_000_000.0
EXPENSE_DAILY = 0.0095 / 252

TICKERS = ["JNJ", "ABT", "BMY", "BDX", "UNH", "MRK", "PG", "KMB", "CLX",
           "GIS", "HSY", "HRL", "PEP", "CL", "MCD", "GD",
           "TJX", "ROST", "AZO", "ORLY", "NKE", "SYY", "CAH", "ITW", "DHR",
           "LOW", "CVS", "TGT", "DLTR", "CHD", "SBUX", "SYK",
           "PFE", "KO", "GPC", "SPY"]

print("downloading daily adjusted closes...")
px = yf.download(TICKERS, start="1999-12-20", end="2014-01-10",
                 interval="1d", auto_adjust=True, progress=False)["Close"]
px = px.ffill()
days = px.loc[START:END].index

# --- 3-month T-bill (financing rate), era-accurate monthly table ---------
# Source: FRED TB3MS monthly averages, %/yr. Linear steps by month.
TBILL = {
    2000: [5.32, 5.55, 5.69, 5.66, 5.79, 5.69, 5.96, 6.09, 6.00, 6.11, 6.17, 5.77],
    2001: [5.15, 4.88, 4.42, 3.87, 3.62, 3.49, 3.51, 3.36, 2.64, 2.16, 1.87, 1.69],
    2002: [1.65, 1.73, 1.79, 1.72, 1.73, 1.70, 1.68, 1.62, 1.63, 1.58, 1.23, 1.19],
    2003: [1.17, 1.17, 1.13, 1.13, 1.07, 0.92, 0.90, 0.95, 0.94, 0.92, 0.93, 0.90],
    2004: [0.88, 0.93, 0.94, 0.94, 1.02, 1.27, 1.33, 1.48, 1.65, 1.76, 2.07, 2.19],
    2005: [2.33, 2.54, 2.74, 2.78, 2.84, 2.97, 3.22, 3.44, 3.42, 3.71, 3.88, 3.89],
    2006: [4.24, 4.43, 4.51, 4.60, 4.72, 4.79, 4.95, 4.96, 4.81, 4.92, 4.94, 4.85],
    2007: [4.98, 5.03, 4.94, 4.87, 4.73, 4.61, 4.82, 4.20, 3.89, 3.90, 3.27, 3.00],
    2008: [2.75, 2.12, 1.26, 1.29, 1.73, 1.86, 1.63, 1.72, 1.13, 0.67, 0.19, 0.03],
    2009: [0.13, 0.30, 0.22, 0.16, 0.18, 0.18, 0.18, 0.17, 0.12, 0.07, 0.05, 0.05],
    2010: [0.06, 0.11, 0.15, 0.16, 0.16, 0.12, 0.16, 0.16, 0.15, 0.13, 0.14, 0.14],
    2011: [0.15, 0.13, 0.10, 0.06, 0.04, 0.04, 0.04, 0.02, 0.01, 0.02, 0.01, 0.01],
    2012: [0.03, 0.09, 0.08, 0.08, 0.09, 0.09, 0.10, 0.10, 0.11, 0.10, 0.09, 0.07],
    2013: [0.07, 0.10, 0.09, 0.06, 0.04, 0.05, 0.04, 0.04, 0.02, 0.05, 0.07, 0.07],
}
rf_daily = pd.Series([TBILL[d.year][d.month - 1] / 100 / 252 for d in days],
                     index=days)

# --- replay the trade log on daily closes -> daily share ledger ----------
trades = json.load(open("trades.json"))


def daily_equity(acct):
    tl = sorted(trades[acct]["trades"], key=lambda x: x["date"])
    sh = {}
    cash = INITIAL
    ti = 0
    eq = np.empty(len(days))
    for i, d in enumerate(days):
        while ti < len(tl) and pd.Timestamp(tl[ti]["date"]) <= d:
            tr = tl[ti]
            t = tr["ticker"]
            p = px.at[d, t]  # execute at current daily close (weekly px ~ same wk)
            if tr["action"] == "SELL":
                cash += sh.pop(t, 0.0) * p
            elif tr["action"] == "BUY_REPL":
                spend = min(tr["amount"], cash)
                sh[t] = sh.get(t, 0.0) + spend / p
                cash -= spend
            else:  # BUY / ADD
                spend = min(tr["amount"], cash)
                sh[t] = sh.get(t, 0.0) + spend / p
                cash -= spend
            ti += 1
        eq[i] = cash + sum(s * px.at[d, t] for t, s in sh.items())
    return pd.Series(eq, index=days, name=acct)


print("replaying trade logs daily...")
base = {"defensive": daily_equity("defensive"),
        "growth": daily_equity("growth"),
        "spy": px.loc[days, "SPY"] / px.at[days[0], "SPY"] * INITIAL}

yrs = (days[-1] - days[0]).days / 365.25
for k, s in base.items():
    print(f"  {k:10} daily-replay CAGR {((s.iloc[-1]/INITIAL)**(1/yrs)-1)*100:5.2f}% "
          f"(weekly engine reference: def 11.02 / gro 17.16 / spy 3.56)")


def lever(series, Y):
    r = series.pct_change().fillna(0.0).values
    rf = rf_daily.values
    e = np.empty(len(r))
    e[0] = INITIAL
    peak = INITIAL
    min_frac_of_peak = 1.0
    for i in range(1, len(r)):
        growth = 1 + Y * r[i] - (Y - 1) * rf[i] - EXPENSE_DAILY
        e[i] = max(e[i - 1] * growth, 0.0)
        if e[i] > peak:
            peak = e[i]
        frac = e[i] / peak if peak > 0 else 0.0
        min_frac_of_peak = min(min_frac_of_peak, frac)
        if e[i] <= 0.0:
            e[i:] = 0.0
            min_frac_of_peak = 0.0
            break
    ser = pd.Series(e, index=series.index)
    maxdd = 1 - min_frac_of_peak
    return ser, maxdd


LEVS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
out = {}
curves = {}
for acct in ("defensive", "growth", "spy"):
    out[acct] = []
    for Y in LEVS:
        ser, maxdd = lever(base[acct], Y)
        final = ser.iloc[-1]
        blown = final <= 0.0
        disq = blown or maxdd >= 0.99
        cagr = ((final / INITIAL) ** (1 / yrs) - 1) * 100 if final > 0 else -100.0
        out[acct].append({
            "leverage": Y, "final_value": round(final),
            "cagr_pct": round(cagr, 2),
            "total_return_pct": round((final / INITIAL - 1) * 100, 1),
            "max_drawdown_pct": round(maxdd * 100, 1),
            "blown": bool(blown), "disqualified": bool(disq)})
        curves[f"{acct}_{Y}"] = ser
        flag = "💥 BLOWN" if blown else ("❌ DISQ (>99% dd)" if disq else "")
        print(f"{acct:10} {Y:.1f}x  final ${final:>13,.0f}  CAGR {cagr:7.2f}%  "
              f"maxDD {maxdd*100:5.1f}%  {flag}")

# best qualifying per account
for acct in out:
    ok = [r for r in out[acct] if not r["disqualified"]]
    best = max(ok, key=lambda r: r["cagr_pct"]) if ok else None
    out[acct + "_best"] = best
    print(f"BEST {acct}: {best['leverage']}x -> {best['cagr_pct']}% CAGR"
          if best else f"BEST {acct}: none qualify")

json.dump(out, open("leverage_results.json", "w"), indent=1)
pd.DataFrame({k: v for k, v in curves.items()}).to_csv("leverage_curves.csv")
base_df = pd.DataFrame(base)
base_df.to_csv("daily_equity.csv")
print("saved leverage_results.json / leverage_curves.csv / daily_equity.csv")
