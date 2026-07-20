"""Correction-sniping v2 — accurate fills, NO time limit, ideal-parameter sweeps.

Accuracy upgrades over v1:
  * Entry is a LIMIT fill at exactly trough*(1+offset): on the first day the
    daily close crosses the trigger, the position starts at the trigger level,
    so the rest of that day's move (trigger -> close) accrues at 5x. (v1
    wrongly entered at the next close, giving away the day-0 pop.)
  * Take-profit is a LIMIT fill at exactly the roof multiple: the fund NAV
    trades continuously, so the day the close-multiple crosses the roof, the
    exit fills at exactly roof (no overshoot credited, none lost).
  * Exit "end" = data end (2013-12-31) if the roof is never reached.
  * Financing (Y-1)*T-bill accrues daily from day 1; 0.95%/yr ER; equity
    floored at 0 (UPRO-style, max loss 100%).
  * Compounded mode is overlap-aware: while a trade is open, later correction
    triggers are SKIPPED (the money is busy) — no double-counting.
Configs tested per account (defensive / growth) and entry offset (+10%, +5%):
  * user config: TP +200%, no time limit
  * TP sweep (no time limit)  -> ideal roof
  * time-limit sweep at TP +200% -> ideal time limit
  * full grid roof x timelimit -> global ideal (fresh & compounded)
"""
import json
import numpy as np
import pandas as pd

TRADE = 100_000.0
Y = 5.0
ER_D = 0.0095 / 252

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
DAYS = base.index
RF = np.array([TBILL[d.year][d.month - 1] / 100 / 252 for d in DAYS])


def find_corrections(s, thresh=0.07):
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


def snipe(series, trough_d, offset, roof_mult, max_days):
    """One trade. Returns dict or None if never triggered.
    Entry limit-fills at trough*(1+offset); exit limit-fills at roof_mult."""
    v = series.values
    ti = series.index.get_loc(trough_d)
    trigger = v[ti] * (1 + offset)
    ei = None
    for i in range(ti, len(v)):          # trough day itself can cross
        if v[i] >= trigger:
            ei = i
            break
    if ei is None:
        return None
    entry_d = DAYS[ei]
    deadline = entry_d + pd.Timedelta(days=max_days) if max_days else None
    # day 0: from trigger fill to close, levered, no financing (intra-day)
    m = 1 + Y * (v[ei] / trigger - 1)
    m = max(m, 0.0)
    mmin = m
    if m >= roof_mult:                    # roof crossed on entry day
        return _mk(trough_d, entry_d, entry_d, "roof", roof_mult, mmin, offset)
    for i in range(ei + 1, len(v)):
        r = v[i] / v[i - 1] - 1
        m = max(m * (1 + Y * r - (Y - 1) * RF[i] - ER_D), 0.0)
        if m >= roof_mult:                # limit TP fills at exactly roof
            return _mk(trough_d, entry_d, DAYS[i], "roof", roof_mult, mmin, offset)
        mmin = min(mmin, m)
        if m <= 0.0:
            return _mk(trough_d, entry_d, DAYS[i], "blown", 0.0, 0.0, offset)
        if deadline is not None and DAYS[i] >= deadline:
            return _mk(trough_d, entry_d, DAYS[i], "time", m, mmin, offset)
    return _mk(trough_d, entry_d, DAYS[-1], "end", m, mmin, offset)


def _mk(trough_d, entry_d, exit_d, reason, mult, mmin, offset):
    return {"trough": str(trough_d.date()), "entry": str(entry_d.date()),
            "exit": str(exit_d.date()), "days_held": (exit_d - entry_d).days,
            "exit_reason": reason, "multiple": round(float(mult), 4),
            "min_mult": round(float(mmin), 4),
            "profit_on_100k": round(TRADE * (mult - 1))}


def run_config(series, offset, roof_pct, max_days):
    corrs = find_corrections(series)
    roof = 1 + roof_pct / 100
    trades = [t for t in (snipe(series, td, offset, roof, max_days)
                          for _, td, _ in corrs) if t]
    # fresh-money stats
    total = sum(t["profit_on_100k"] for t in trades)
    # overlap-aware compounding: skip trades that trigger while one is open
    comp = 1.0
    busy_until = None
    comp_trades = 0
    for t in trades:
        e = pd.Timestamp(t["entry"])
        if busy_until is not None and e <= busy_until:
            continue
        comp *= t["multiple"]
        busy_until = pd.Timestamp(t["exit"])
        comp_trades += 1
        if comp <= 0:
            break
    return trades, {
        "offset_pct": int(offset * 100), "roof_pct": roof_pct,
        "max_days": max_days, "n_trades": len(trades),
        "wins": sum(1 for t in trades if t["multiple"] > 1),
        "roof_hits": sum(1 for t in trades if t["exit_reason"] == "roof"),
        "blowups": sum(1 for t in trades if t["exit_reason"] == "blown"),
        "worst_multiple": round(min((t["multiple"] for t in trades), default=1), 3),
        "avg_days_held": round(float(np.mean([t["days_held"] for t in trades])))
        if trades else 0,
        "total_profit_fresh": round(total),
        "compounded_final": round(TRADE * comp),
        "compounded_trades_used": comp_trades}


ROOFS = list(range(25, 401, 25)) + list(range(450, 1001, 50))
TLIMITS = [30, 60, 90, 106, 120, 150, 180, 240, 300, 365, 500, 730, 1095, None]

out = {}
for acct in ("defensive", "growth"):
    s = base[acct]
    out[acct] = {}
    for offset in (0.10, 0.05):
        key = f"off{int(offset*100)}"
        o = {}
        # --- user config: TP 200, no limit -----------------------------
        trades, sm = run_config(s, offset, 200, None)
        o["user_tp200_nolimit"] = {"summary": sm, "trades": trades}
        # --- roof sweep, no limit --------------------------------------
        o["roof_sweep_nolimit"] = [run_config(s, offset, r, None)[1]
                                   for r in ROOFS]
        # --- time-limit sweep at TP 200 --------------------------------
        o["tlimit_sweep_tp200"] = [run_config(s, offset, 200, tl)[1]
                                   for tl in TLIMITS]
        # --- full grid ---------------------------------------------------
        grid = [run_config(s, offset, r, tl)[1]
                for r in ROOFS for tl in TLIMITS]
        o["ideal_fresh"] = max(grid, key=lambda x: x["total_profit_fresh"])
        o["ideal_compounded"] = max(grid, key=lambda x: x["compounded_final"])
        # keep ideal trades for reporting
        o["ideal_fresh_trades"] = run_config(
            s, offset, o["ideal_fresh"]["roof_pct"],
            o["ideal_fresh"]["max_days"])[0]
        out[acct][key] = o
        sm = o["user_tp200_nolimit"]["summary"]
        print(f"{acct:10} +{int(offset*100)}%: TP200/nolimit "
              f"{sm['n_trades']}tr {sm['roof_hits']}roof {sm['blowups']}blown "
              f"fresh ${sm['total_profit_fresh']:,} comp ${sm['compounded_final']:,}")
        print(f"           ideal fresh: roof {o['ideal_fresh']['roof_pct']}% "
              f"tl {o['ideal_fresh']['max_days']} -> "
              f"${o['ideal_fresh']['total_profit_fresh']:,}")
        print(f"           ideal comp:  roof {o['ideal_compounded']['roof_pct']}% "
              f"tl {o['ideal_compounded']['max_days']} -> "
              f"${o['ideal_compounded']['compounded_final']:,}")

json.dump(out, open("correction_snipe2.json", "w"), indent=1)
print("saved correction_snipe2.json")
