"""Find the ideal profit target for the continuously-compounded correction
sniper: the roof-vs-corrections-caught tradeoff, fine-grained.

One pot, rolled trade to trade. While deployed, new correction triggers are
skipped — so a higher roof that keeps the pot stuck through the next
correction(s) automatically pays for it. Fine sweep: roof 25%..1000% step 5%,
no time limit, both accounts, entries bottom+5% and bottom+10%.
Also cross-checks roof x time-limit grid (fine roofs x 14 limits) in case a
time-stop beats pure-roof (it can: it frees the pot before long dead zones).
Reuses the accurate v2 fill mechanics (limit entry at trough*(1+off), limit
TP at exactly the roof, era T-bill financing, 0.95% ER, floor at 0).
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


def entry_index(v, ti, trigger):
    for i in range(ti, len(v)):
        if v[i] >= trigger:
            return i
    return None


def compound_run(series, offset, roof_pct, max_days=None):
    """One pot rolled through all corrections (overlap-aware).
    Returns (final_pot, legs) — legs = list of dicts."""
    v = series.values
    roof = 1 + roof_pct / 100
    corrs = find_corrections(series)
    pot = TRADE
    busy_until_i = -1
    legs = []
    for _, trough_d, depth in corrs:
        ti = series.index.get_loc(trough_d)
        trigger = v[ti] * (1 + offset)
        ei = entry_index(v, ti, trigger)
        if ei is None or ei <= busy_until_i:
            continue
        # simulate leg
        m = max(1 + Y * (v[ei] / trigger - 1), 0.0)
        mmin = m
        exit_i, reason, mult = None, None, None
        if m >= roof:
            exit_i, reason, mult = ei, "roof", roof
        else:
            deadline = (DAYS[ei] + pd.Timedelta(days=max_days)
                        if max_days else None)
            for i in range(ei + 1, len(v)):
                r = v[i] / v[i - 1] - 1
                m = max(m * (1 + Y * r - (Y - 1) * RF[i] - ER_D), 0.0)
                mmin = min(mmin, m)
                if m >= roof:
                    exit_i, reason, mult = i, "roof", roof
                    break
                if m <= 0.0:
                    exit_i, reason, mult = i, "blown", 0.0
                    break
                if deadline is not None and DAYS[i] >= deadline:
                    exit_i, reason, mult = i, "time", m
                    break
            if exit_i is None:
                exit_i, reason, mult = len(v) - 1, "end", m
        pot *= mult
        legs.append({"trough": str(trough_d.date()),
                     "entry": str(DAYS[ei].date()),
                     "exit": str(DAYS[exit_i].date()),
                     "days": int((DAYS[exit_i] - DAYS[ei]).days),
                     "reason": reason, "mult": round(float(mult), 4),
                     "min_mult": round(float(mmin), 4),
                     "pot_after": round(pot)})
        busy_until_i = exit_i
        if pot <= 0:
            break
    return pot, legs


FINE = list(range(25, 1001, 5))
TL = [60, 90, 120, 150, 180, 240, 300, 365, 500, 730, 1095, None]

out = {}
for acct in ("defensive", "growth"):
    s = base[acct]
    out[acct] = {}
    for offset in (0.10, 0.05):
        key = f"off{int(offset*100)}"
        sweep = []
        for r in FINE:
            pot, legs = compound_run(s, offset, r)
            sweep.append({"roof_pct": r, "final": round(pot),
                          "legs": len(legs)})
        best = max(sweep, key=lambda x: x["final"])
        _, best_legs = compound_run(s, offset, best["roof_pct"])
        # grid cross-check with time limits (coarser roofs for speed: step 25)
        grid_best = None
        for r in range(25, 1001, 25):
            for tl in TL:
                pot, _ = compound_run(s, offset, r, tl)
                if grid_best is None or pot > grid_best[0]:
                    grid_best = (pot, r, tl)
        gpot, gr, gtl = grid_best
        _, glegs = compound_run(s, offset, gr, gtl)
        out[acct][key] = {
            "sweep": sweep,
            "best_roof_only": {"roof_pct": best["roof_pct"],
                               "final": best["final"],
                               "legs": best_legs},
            "best_with_timelimit": {"roof_pct": gr, "max_days": gtl,
                                    "final": round(gpot), "legs": glegs}}
        yrs = 14.0
        print(f"{acct:10} +{int(offset*100)}%: BEST roof-only "
              f"{best['roof_pct']}% -> ${best['final']:,} "
              f"({len(best_legs)} legs, CAGR "
              f"{((best['final']/TRADE)**(1/yrs)-1)*100:.1f}%)")
        print(f"{'':14}BEST w/ time-limit roof {gr}% tl {gtl} -> "
              f"${gpot:,.0f} ({len(glegs)} legs, CAGR "
              f"{((gpot/TRADE)**(1/yrs)-1)*100:.1f}%)")

json.dump(out, open("snipe_optimal.json", "w"), indent=1)
print("saved snipe_optimal.json")
