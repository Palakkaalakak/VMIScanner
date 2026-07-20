"""VMI 2000-2013 backtest v2 — TWO accounts, DCF-gated entries.

Standing in Jan 2000. ZERO dotcom/tech/telecom/internet exposure.
Account A (DEFENSIVE): healthcare + staples + defensive franchises.
Account B (GROWTH): higher-growth great businesses (retail concepts,
medtech, distribution, industrial compounders) — still non-dotcom,
still DCF-undervalued before buying.

DCF (course-style 20y, VMI Lesson 5 / StockOracle structure):
  base flow/share ~ trailing EPS x OCF-conversion multiple
  years 1-10 grow at g (late-1990s consensus-proxy growth, knowable then)
  years 11-20 grow at 4%; NO terminal value
  discount r = rf + beta*MRP with rf = 6.5% (Jan-2000 10y Treasury), MRP 4%
  IV/price at anchor = factor(g,r) x ocf_mult / trailing P/E — scale-
  invariant, so it works directly on the adjusted price series.
  IV refreshed annually: grows at min(g, 10%)/yr (conservative).

BUY: tranche 1 the first week price < IV. Adds (max 3 tranches = 6.25%
cap) only when price <= 40w SMA (weekly-TF support) AND price < IV,
min 8 weeks apart. SELL only for fraud/illegal/business deterioration:
  A: BMY 2002-07 channel-stuffing accounting scandal -> PFE
     UNH 2006-10 options-backdating scandal -> KO
  B: CAH 2004-07 SEC accounting probe (revenue classification) -> GPC
MRK held through Vioxx (product withdrawal, not fraud; checklist intact).
Trailing P/E / growth / beta / OCF-conversion are documented-era
approximations (e.g. the March-2000 old-economy valuation trough).
"""
import json
import numpy as np
import pandas as pd

INITIAL = 1_000_000.0
N = 16
CAP = INITIAL / N
TRANCHE = CAP / 3.0
GAP = 8
RF, MRP = 0.065, 0.04
START, END = "2000-01-01", "2013-12-31"

#          ticker: (PE_jan2000, g_yrs1_10, beta, ocf_mult)
DEFENSIVE = {
    "JNJ": (29, .125, .85, 1.15), "ABT": (21, .12, .80, 1.20),
    "BMY": (28, .12, .85, 1.10), "BDX": (19, .11, .75, 1.35),
    "UNH": (16, .15, .90, 1.30), "MRK": (28, .12, .90, 1.10),
    "PG":  (38, .11, .75, 1.25), "KMB": (20, .10, .80, 1.30),
    "CLX": (30, .11, .70, 1.25), "GIS": (17, .09, .65, 1.20),
    "HSY": (22, .10, .70, 1.20), "HRL": (16, .10, .65, 1.25),
    "PEP": (26, .11, .80, 1.30), "CL":  (33, .12, .80, 1.25),
    "MCD": (27, .11, .85, 1.50), "GD":  (12, .10, .75, 1.20),
}
GROWTH = {
    "TJX": (14, .15, .90, 1.30), "ROST": (12, .15, .95, 1.25),
    "AZO": (11, .14, .90, 1.30), "ORLY": (22, .22, .95, 1.20),
    "NKE": (22, .14, 1.00, 1.20), "SYY": (27, .14, .75, 1.25),
    "CAH": (22, .20, .90, 1.10), "ITW": (19, .14, .95, 1.20),
    "DHR": (24, .17, .95, 1.30), "LOW": (21, .22, 1.05, 1.20),
    "CVS": (23, .15, .85, 1.20), "TGT": (22, .15, .95, 1.40),
    "DLTR": (28, .25, 1.00, 1.20), "CHD": (18, .12, .70, 1.25),
    "SBUX": (42, .27, 1.10, 1.60), "SYK": (32, .20, .90, 1.30),
}
REPL = {"A": [("2002-07-08", "BMY", "PFE"), ("2006-10-16", "UNH", "KO")],
        "B": [("2004-07-12", "CAH", "GPC")]}


def dcf_factor(g, beta):
    """PV of 20y of $1-anchored flows: g for yrs 1-10, 4% yrs 11-20."""
    r = RF + beta * MRP
    f = sum((1 + g) ** t / (1 + r) ** t for t in range(1, 11))
    y10 = (1 + g) ** 10
    f += sum(y10 * 1.04 ** k / (1 + r) ** (10 + k) for k in range(1, 11))
    return f


px = pd.read_csv("weekly_adj.csv", index_col=0, parse_dates=True)
px = px.loc["1999-01-01":END]
sma40 = px.rolling(40).mean()
dates = px.loc[START:END].index
anchor = dates[0]


def run_account(book, repls, label):
    iv0 = {t: px.at[anchor, t] * dcf_factor(g, b) * m / pe
           for t, (pe, g, b, m) in book.items()}
    g_iv = {t: min(book[t][1], 0.10) for t in book}

    def iv_at(t, d):
        return iv0[t] * (1 + g_iv[t]) ** ((d - anchor).days / 365.25)

    sh, tr, last = {}, {}, {}
    cash = INITIAL
    log = []
    repl = {pd.Timestamp(d): (o, n) for d, o, n in repls}
    eq = []
    for i, d in enumerate(dates):
        # scandal sells -> immediate full redeploy into replacement
        for rd, (old, new) in list(repl.items()):
            if d >= rd and old in sh:
                proceeds = sh.pop(old) * px.at[d, old]
                tr.pop(old); last.pop(old)
                sh[new] = proceeds / px.at[d, new]
                tr[new] = 3; last[new] = i
                log.append((d.date(),
                            f"SELL {old} (scandal/probe) ${proceeds:,.0f} -> BUY {new}"))
                del repl[rd]
        for t in book:
            p = px.at[d, t] if t in px.columns else np.nan
            if np.isnan(p):
                continue
            if t not in tr:  # first entry gated purely by DCF undervaluation
                if p < iv_at(t, d) and cash >= TRANCHE:
                    sh[t] = TRANCHE / p; tr[t] = 1; last[t] = i
                    cash -= TRANCHE
                    log.append((d.date(),
                                f"BUY {t} T1 ${TRANCHE:,.0f} @ {p:.2f} "
                                f"({p / iv_at(t, d) * 100:.0f}% of IV {iv_at(t, d):.2f})"))
            elif tr[t] < 3 and i - last[t] >= GAP:
                s = sma40.at[d, t]
                if not np.isnan(s) and p <= s * 1.01 and p < iv_at(t, d) \
                        and cash >= TRANCHE:
                    sh[t] += TRANCHE / p; tr[t] += 1; last[t] = i
                    cash -= TRANCHE
                    log.append((d.date(),
                                f"ADD {t} T{tr[t]} ${TRANCHE:,.0f} @ {p:.2f} "
                                f"(40wSMA support + under IV)"))
        eq.append(cash + sum(s_ * px.at[d, t] for t, s_ in sh.items()))
    return pd.Series(eq, index=dates, name=label), log, {t: tr.get(t, 0) for t in book}


def drawdowns(series, thresh=0.07):
    peak, peak_d = series.iloc[0], series.index[0]
    trough, trough_d, in_dd, out = peak, peak_d, False, []
    for d, v in series.items():
        if v >= peak:
            if in_dd and (peak - trough) / peak >= thresh:
                out.append((peak_d, trough_d, d, (peak - trough) / peak))
            peak, peak_d, in_dd = v, d, False
            trough, trough_d = v, d
        else:
            in_dd = True
            if v < trough:
                trough, trough_d = v, d
    if in_dd and (peak - trough) / peak >= thresh:
        out.append((peak_d, trough_d, None, (peak - trough) / peak))
    return out


eqA, logA, trA = run_account(DEFENSIVE, REPL["A"], "Defensive")
eqB, logB, trB = run_account(GROWTH, REPL["B"], "Growth")
spy = px.loc[dates, "SPY"]
spy_eq = spy / spy.iloc[0] * INITIAL

yrs = (dates[-1] - dates[0]).days / 365.25
out = {}
for name, ser in (("defensive", eqA), ("growth", eqB), ("spy", spy_eq)):
    out[name] = {"total_return_pct": round((ser.iloc[-1] / INITIAL - 1) * 100, 1),
                 "cagr_pct": round(((ser.iloc[-1] / INITIAL) ** (1 / yrs) - 1) * 100, 2),
                 "final_value": round(ser.iloc[-1])}
    out[name]["corrections_7pct_plus"] = [
        {"peak": str(pk.date()), "trough": str(tg.date()),
         "recovered": str(rc.date()) if rc is not None else None,
         "depth_pct": round(dep * 100, 1),
         "decline_days": (tg - pk).days,
         "recovery_days_from_trough": (rc - tg).days if rc is not None else None,
         "total_underwater_days": (rc - pk).days if rc is not None else None}
        for pk, tg, rc, dep in drawdowns(ser)]

eqA.to_csv("eq_defensive.csv"); eqB.to_csv("eq_growth.csv")
spy_eq.to_csv("eq_spy.csv")
json.dump(out, open("stats2.json", "w"), indent=1)
with open("trades_defensive.txt", "w") as f:
    f.writelines(f"{d}  {m}\n" for d, m in logA)
with open("trades_growth.txt", "w") as f:
    f.writelines(f"{d}  {m}\n" for d, m in logB)

print("=== DCF at Jan 2000: IV/price (>1.00 = undervalued at anchor) ===")
for label, book in (("DEF", DEFENSIVE), ("GRO", GROWTH)):
    for t, (pe, g, b, m) in sorted(book.items()):
        print(f"  {label} {t:5} PE {pe:3}  g {g*100:4.1f}%  IV/P {dcf_factor(g, b) * m / pe:5.2f}")
print()
for name in ("defensive", "growth", "spy"):
    s = out[name]
    print(f"{name:10}: total +{s['total_return_pct']}%  CAGR {s['cagr_pct']}%  "
          f"final ${s['final_value']:,}  corrections(7%+): "
          f"{len(s['corrections_7pct_plus'])}")
print("\nDEF tranches:", trA)
print("GRO tranches:", trB)
