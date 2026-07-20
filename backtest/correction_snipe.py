"""Correction-sniping study: enter a 5x daily-rebalanced (UPRO-style)
version of the VMI portfolio at bottom+10% of every 7%+ correction.

Entry: first daily close >= trough*1.10 after the trough of each 7%+
correction (corrections detected on the DAILY unlevered equity curve,
same 7% peak-trough rule as the main backtest).
Position: $100k into the 5x levered portfolio (daily rebalance,
financing (Y-1)*T-bill + 0.95%/yr ER, floor at 0 = 100% max loss).
Exit: profit roof hit (+200% => 3x multiple) OR 3.5 months (106 days) pass.
Also tested: +300%, +400% roofs, plus a roof sweep to find the ideal.
Trades are per-correction and independent ($100k fresh each time);
"compounded" also reported (roll the full proceeds into the next trade).
"""
import json
import numpy as np
import pandas as pd

INITIAL_TRADE = 100_000.0
Y = 5.0
EXPENSE_DAILY = 0.0095 / 252
HOLD_DAYS = 106  # 3.5 months in calendar days

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

base = pd.read_csv("daily_equity.csv", index_col=0, parse_dates=True)
days = base.index
rf_daily = pd.Series([TBILL[d.year][d.month - 1] / 100 / 252 for d in days],
                     index=days)


def find_corrections(s, thresh=0.07):
    """Peak/trough pairs with >=7% depth on the daily series."""
    peak, peak_d = s.iloc[0], s.index[0]
    trough, trough_d, in_dd = peak, peak_d, False
    out = []
    for d, v in s.items():
        if v >= peak:
            if in_dd and (peak - trough) / peak >= thresh:
                out.append((peak_d, trough_d, (peak - trough) / peak))
            peak, peak_d, in_dd = v, d, False
            trough, trough_d = v, d
        else:
            in_dd = True
            if v < trough:
                trough, trough_d = v, d
    if in_dd and (peak - trough) / peak >= thresh:
        out.append((peak_d, trough_d, (peak - trough) / peak))
    return out


def lever_path(series, start_idx):
    """5x daily-rebalanced growth multiples from start_idx onward."""
    r = series.pct_change().fillna(0.0).values
    rf = rf_daily.values
    n = len(r)
    mult = np.empty(n - start_idx)
    mult[0] = 1.0
    for j in range(1, n - start_idx):
        i = start_idx + j
        g = 1 + Y * r[i] - (Y - 1) * rf[i] - EXPENSE_DAILY
        mult[j] = max(mult[j - 1] * g, 0.0)
        if mult[j] <= 0.0:
            mult[j:] = 0.0
            break
    return mult


def run_trades(series, roof_mult):
    """One trade per correction: enter at first close >= trough*1.10 after
    trough; exit at roof multiple or after HOLD_DAYS calendar days."""
    corrs = find_corrections(series)
    trades = []
    for peak_d, trough_d, depth in corrs:
        trough_px = series.loc[trough_d]
        after = series.loc[trough_d:]
        hit = after[after >= trough_px * 1.10]
        if hit.empty:
            continue
        entry_d = hit.index[0]
        i0 = series.index.get_loc(entry_d)
        mult = lever_path(series, i0)
        idx = series.index[i0:]
        deadline = entry_d + pd.Timedelta(days=HOLD_DAYS)
        exit_j = None
        reason = "time"
        for j in range(1, len(mult)):
            if mult[j] >= roof_mult:
                exit_j, reason = j, "roof"
                break
            if mult[j] <= 0.0:
                exit_j, reason = j, "blown"
                break
            if idx[j] >= deadline:
                exit_j = j
                break
        if exit_j is None:  # data ends first
            exit_j = len(mult) - 1
            reason = "end"
        m = mult[exit_j]
        trades.append({
            "correction_peak": str(peak_d.date()),
            "trough": str(trough_d.date()), "depth_pct": round(depth * 100, 1),
            "entry": str(entry_d.date()), "exit": str(idx[exit_j].date()),
            "days_held": (idx[exit_j] - entry_d).days,
            "exit_reason": reason, "multiple": round(m, 3),
            "profit_on_100k": round(INITIAL_TRADE * (m - 1)),
            "min_mult_during": round(float(mult[:exit_j + 1].min()), 3),
        })
    return trades


def summarize(trades):
    if not trades:
        return {}
    total = sum(t["profit_on_100k"] for t in trades)
    comp = 1.0
    for t in trades:
        comp *= t["multiple"]
    wins = sum(1 for t in trades if t["multiple"] > 1)
    return {"n_trades": len(trades), "wins": wins,
            "losses": len(trades) - wins,
            "roof_hits": sum(1 for t in trades if t["exit_reason"] == "roof"),
            "blowups": sum(1 for t in trades if t["exit_reason"] == "blown"),
            "total_profit_fresh_100k_each": round(total),
            "avg_multiple": round(float(np.mean([t["multiple"] for t in trades])), 3),
            "worst_multiple": round(min(t["multiple"] for t in trades), 3),
            "compounded_100k_final": round(INITIAL_TRADE * comp)}


out = {}
for acct in ("defensive", "growth"):
    s = base[acct]
    out[acct] = {}
    for roof_pct in (200, 300, 400):
        trades = run_trades(s, 1 + roof_pct / 100)
        out[acct][f"roof_{roof_pct}"] = {"trades": trades,
                                         "summary": summarize(trades)}
        sm = out[acct][f"roof_{roof_pct}"]["summary"]
        print(f"{acct:10} roof +{roof_pct}%: {sm['n_trades']} trades, "
              f"{sm['roof_hits']} roofs hit, {sm['blowups']} blowups, "
              f"fresh-$100k total profit ${sm['total_profit_fresh_100k_each']:,}, "
              f"compounded $100k -> ${sm['compounded_100k_final']:,}")

    # ideal-roof sweep 25% .. 1000%
    sweep = []
    for roof_pct in list(range(25, 401, 25)) + list(range(450, 1001, 50)):
        trades = run_trades(s, 1 + roof_pct / 100)
        sm = summarize(trades)
        sweep.append({"roof_pct": roof_pct,
                      "total_profit": sm.get("total_profit_fresh_100k_each", 0),
                      "compounded": sm.get("compounded_100k_final", 0),
                      "roof_hits": sm.get("roof_hits", 0),
                      "n": sm.get("n_trades", 0)})
    out[acct]["sweep"] = sweep
    best_f = max(sweep, key=lambda x: x["total_profit"])
    best_c = max(sweep, key=lambda x: x["compounded"])
    out[acct]["ideal_roof_fresh"] = best_f
    out[acct]["ideal_roof_compounded"] = best_c
    print(f"{acct:10} IDEAL roof (fresh $100k):  +{best_f['roof_pct']}% "
          f"-> total ${best_f['total_profit']:,}")
    print(f"{acct:10} IDEAL roof (compounded):   +{best_c['roof_pct']}% "
          f"-> ${best_c['compounded']:,}")

json.dump(out, open("correction_snipe.json", "w"), indent=1)
print("saved correction_snipe.json")
