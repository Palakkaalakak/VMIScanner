"""Multi-cohort VMI backtest: pick stocks in 2000/2005/2010/2015/2020 with
era-knowable inputs, run each to July 2026 with the SAME engine/rules as
simulate2.py (DCF-gated entries, 3 tranches, 40w-SMA adds, scandal-only sells).

2000 cohort = the original two books (defensive + growth) EXTENDED to 2026.
2005/2010/2015/2020 = one 16-stock quality book each (the defensive/growth
split in 2000 was the lost-decade hindsight; later cohorts get the standard
VMI quality blend, no sector hindsight).

Era inputs per cohort: trailing P/E at anchor, consensus-proxy growth, beta,
OCF-conversion multiple — documented approximations knowable at the time.
rf = 10y Treasury at anchor: 2000 6.5%, 2005 4.2%, 2010 3.7%, 2015 2.2%,
2020 1.9%. MRP 4% throughout. IV grows at min(g,10%)/yr as before.

Scandal sells (same fraud/deterioration-only rule):
  2000A: BMY 2002-07-08 -> PFE ; UNH 2006-10-16 -> KO
  2000B: CAH 2004-07-12 -> GPC
  2005:  UNH 2006-10-16 (options backdating) -> KO
  2010/2015/2020: UNH 2025-05-15 (DOJ criminal accounting probe over
    Medicare billing; CEO exit) -> replacement listed per book.
"""
import json
import numpy as np
import pandas as pd
import yfinance as yf

INITIAL = 1_000_000.0
N = 16
TRANCHE = INITIAL / N / 3.0
GAP = 8
MRP = 0.04
END = "2026-07-17"

BOOKS_2000_DEF = {
    "JNJ": (29, .125, .85, 1.15), "ABT": (21, .12, .80, 1.20),
    "BMY": (28, .12, .85, 1.10), "BDX": (19, .11, .75, 1.35),
    "UNH": (16, .15, .90, 1.30), "MRK": (28, .12, .90, 1.10),
    "PG":  (38, .11, .75, 1.25), "KMB": (20, .10, .80, 1.30),
    "CLX": (30, .11, .70, 1.25), "GIS": (17, .09, .65, 1.20),
    "HSY": (22, .10, .70, 1.20), "HRL": (16, .10, .65, 1.25),
    "PEP": (26, .11, .80, 1.30), "CL":  (33, .12, .80, 1.25),
    "MCD": (27, .11, .85, 1.50), "GD":  (12, .10, .75, 1.20),
}
BOOKS_2000_GRO = {
    "TJX": (14, .15, .90, 1.30), "ROST": (12, .15, .95, 1.25),
    "AZO": (11, .14, .90, 1.30), "ORLY": (22, .22, .95, 1.20),
    "NKE": (22, .14, 1.00, 1.20), "SYY": (27, .14, .75, 1.25),
    "CAH": (22, .20, .90, 1.10), "ITW": (19, .14, .95, 1.20),
    "DHR": (24, .17, .95, 1.30), "LOW": (21, .22, 1.05, 1.20),
    "CVS": (23, .15, .85, 1.20), "TGT": (22, .15, .95, 1.40),
    "DLTR": (28, .25, 1.00, 1.20), "CHD": (18, .12, .70, 1.25),
    "SBUX": (42, .27, 1.10, 1.60), "SYK": (32, .20, .90, 1.30),
}
BOOK_2005 = {  # Jan-2005 era inputs
    "MSFT": (24, .12, .95, 1.40), "JNJ": (21, .11, .70, 1.15),
    "PG":  (21, .11, .65, 1.25), "KO":  (21, .09, .70, 1.25),
    "PEP": (22, .11, .70, 1.25), "WMT": (22, .12, .75, 1.30),
    "MCD": (17, .09, .80, 1.60), "UNH": (20, .18, .85, 1.30),
    "SYK": (30, .20, .90, 1.30), "DHR": (22, .15, .95, 1.30),
    "TJX": (19, .13, .85, 1.30), "ROST": (19, .13, .90, 1.25),
    "ORLY": (21, .18, .90, 1.20), "AZO": (13, .12, .85, 1.30),
    "CVS": (17, .13, .80, 1.20), "BDX": (20, .11, .70, 1.35),
}
BOOK_2010 = {  # Jan-2010 era inputs (post-GFC valuations)
    "AAPL": (21, .20, 1.10, 1.30), "MSFT": (17, .11, .95, 1.35),
    "GOOGL": (24, .20, 1.05, 1.35), "V": (26, .18, .90, 1.30),
    "MA":  (20, .18, .95, 1.25), "JNJ": (14, .08, .60, 1.15),
    "PG":  (16, .09, .60, 1.30), "KO":  (19, .09, .60, 1.25),
    "MCD": (16, .10, .65, 1.50), "NKE": (17, .12, .90, 1.25),
    "TJX": (15, .13, .80, 1.30), "ROST": (15, .13, .85, 1.25),
    "ORLY": (19, .16, .85, 1.35), "AZO": (13, .13, .75, 1.30),
    "DHR": (20, .14, .95, 1.30), "UNH": (9, .11, .80, 1.30),
}
BOOK_2015 = {  # Jan-2015 era inputs
    "AAPL": (17, .13, .95, 1.30), "MSFT": (18, .10, .95, 1.40),
    "GOOGL": (25, .16, 1.00, 1.40), "V": (30, .16, .95, 1.35),
    "MA":  (28, .16, 1.00, 1.30), "UNH": (18, .13, .80, 1.30),
    "HD":  (23, .13, .95, 1.30), "COST": (29, .11, .80, 1.50),
    "NKE": (28, .13, .90, 1.25), "SBUX": (30, .17, .85, 1.50),
    "ORLY": (26, .15, .85, 1.30), "AZO": (19, .12, .70, 1.30),
    "TJX": (21, .11, .75, 1.30), "ROST": (21, .12, .80, 1.25),
    "DHR": (22, .12, .95, 1.30), "DIS": (22, .12, 1.00, 1.30),
}
BOOK_2020 = {  # Jan-2020 era inputs
    "AAPL": (24, .12, 1.10, 1.35), "MSFT": (30, .15, .95, 1.45),
    "GOOGL": (27, .16, 1.00, 1.40), "V": (34, .15, .95, 1.35),
    "MA":  (38, .17, 1.05, 1.30), "UNH": (19, .13, .85, 1.30),
    "HD":  (22, .10, 1.00, 1.30), "COST": (35, .11, .75, 1.50),
    "NKE": (34, .13, .90, 1.30), "SBUX": (30, .12, .80, 1.50),
    "ORLY": (22, .13, .90, 1.30), "AZO": (18, .11, .80, 1.30),
    "TMO": (26, .13, .95, 1.35), "DHR": (32, .14, .90, 1.35),
    "ADBE": (42, .18, 1.00, 1.45), "NVDA": (40, .20, 1.30, 1.30),
}

COHORTS = {
    "2000_defensive": {"book": BOOKS_2000_DEF, "anchor_year": 2000, "rf": .065,
                       "repl": [("2002-07-08", "BMY", "PFE"),
                                ("2006-10-16", "UNH", "KO")]},
    "2000_growth": {"book": BOOKS_2000_GRO, "anchor_year": 2000, "rf": .065,
                    "repl": [("2004-07-12", "CAH", "GPC")]},
    "2005": {"book": BOOK_2005, "anchor_year": 2005, "rf": .042,
             "repl": [("2006-10-16", "UNH", "KO")]},
    "2010": {"book": BOOK_2010, "anchor_year": 2010, "rf": .037,
             "repl": [("2025-05-15", "UNH", "COST")]},
    "2015": {"book": BOOK_2015, "anchor_year": 2015, "rf": .022,
             "repl": [("2025-05-15", "UNH", "KO")]},
    "2020": {"book": BOOK_2020, "anchor_year": 2020, "rf": .019,
             "repl": [("2025-05-15", "UNH", "JNJ")]},
}

ALL_TICKERS = sorted({t for c in COHORTS.values() for t in c["book"]}
                     | {"PFE", "KO", "GPC", "COST", "JNJ", "SPY"})

print(f"downloading weekly closes for {len(ALL_TICKERS)} tickers 1999->2026...")
px = yf.download(ALL_TICKERS, start="1999-01-01", end=END, interval="1wk",
                 auto_adjust=True, progress=False)["Close"].ffill()
px.to_csv("weekly_adj_2026.csv")
sma40 = px.rolling(40).mean()


def dcf_factor(g, beta, rf):
    r = rf + beta * MRP
    f = sum((1 + g) ** t / (1 + r) ** t for t in range(1, 11))
    y10 = (1 + g) ** 10
    f += sum(y10 * 1.04 ** k / (1 + r) ** (10 + k) for k in range(1, 11))
    return f


def run(book, repls, anchor_year, rf):
    dates = px.loc[f"{anchor_year}-01-01":END].index
    anchor = dates[0]
    iv0 = {t: px.at[anchor, t] * dcf_factor(g, b, rf) * m / pe
           for t, (pe, g, b, m) in book.items()}
    g_iv = {t: min(book[t][1], 0.10) for t in book}

    def iv_at(t, d):
        return iv0[t] * (1 + g_iv[t]) ** ((d - anchor).days / 365.25)

    sh, tr, last = {}, {}, {}
    cash = INITIAL
    log = []
    repl = {pd.Timestamp(d): (o, n) for d, o, n in repls}
    ivmap = dict(iv0)
    gmap = dict(g_iv)
    eq = []
    for i, d in enumerate(dates):
        for rd, (old, new) in list(repl.items()):
            if d >= rd and old in sh:
                proceeds = sh.pop(old) * px.at[d, old]
                tr.pop(old, None); last.pop(old, None)
                sh[new] = proceeds / px.at[d, new]
                tr[new] = 3; last[new] = i
                log.append((str(d.date()),
                            f"SELL {old} (scandal) ${proceeds:,.0f} -> {new}"))
                del repl[rd]
        for t in book:
            if t not in px.columns or np.isnan(px.at[d, t]):
                continue
            p = px.at[d, t]
            if t not in tr:
                if p < iv_at(t, d) and cash >= TRANCHE:
                    sh[t] = TRANCHE / p; tr[t] = 1; last[t] = i
                    cash -= TRANCHE
                    log.append((str(d.date()), f"BUY {t} T1 @ {p:.2f}"))
            elif tr[t] < 3 and i - last[t] >= GAP:
                s = sma40.at[d, t]
                if not np.isnan(s) and p <= s * 1.01 and p < iv_at(t, d) \
                        and cash >= TRANCHE:
                    sh[t] += TRANCHE / p; tr[t] += 1; last[t] = i
                    cash -= TRANCHE
                    log.append((str(d.date()), f"ADD {t} T{tr[t]} @ {p:.2f}"))
        eq.append(cash + sum(s_ * px.at[d, t] for t, s_ in sh.items()))
    ser = pd.Series(eq, index=dates)
    n_entered = len([t for t in tr])
    return ser, log, n_entered


def maxdd(s):
    return float((1 - s / s.cummax()).max())


out = {}
curves = {}
for name, cfg in COHORTS.items():
    ser, log, n_ent = run(cfg["book"], cfg["repl"], cfg["anchor_year"], cfg["rf"])
    yrs = (ser.index[-1] - ser.index[0]).days / 365.25
    spy = px.loc[ser.index, "SPY"]
    spy_eq = spy / spy.iloc[0] * INITIAL
    out[name] = {
        "anchor": str(ser.index[0].date()), "end": str(ser.index[-1].date()),
        "years": round(yrs, 1), "stocks_entered": n_ent,
        "final_value": round(ser.iloc[-1]),
        "total_return_pct": round((ser.iloc[-1] / INITIAL - 1) * 100, 1),
        "cagr_pct": round(((ser.iloc[-1] / INITIAL) ** (1 / yrs) - 1) * 100, 2),
        "max_dd_pct": round(maxdd(ser) * 100, 1),
        "spy_final": round(spy_eq.iloc[-1]),
        "spy_cagr_pct": round(((spy_eq.iloc[-1] / INITIAL) ** (1 / yrs) - 1) * 100, 2),
        "spy_max_dd_pct": round(maxdd(spy_eq) * 100, 1),
        "n_trades": len(log), "log": log}
    curves[name] = ser
    curves[f"spy_{name}"] = spy_eq
    r = out[name]
    print(f"{name:15} {r['anchor']}->{r['end']} ({r['years']}y) "
          f"final ${r['final_value']:,} CAGR {r['cagr_pct']}% "
          f"(SPY {r['spy_cagr_pct']}%) maxDD {r['max_dd_pct']}% "
          f"entered {n_ent}/16")

pd.DataFrame(curves).to_csv("cohort_curves.csv")
json.dump(out, open("cohort_results.json", "w"), indent=1)
print("saved cohort_results.json / cohort_curves.csv")
