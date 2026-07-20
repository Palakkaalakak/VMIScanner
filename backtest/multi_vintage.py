"""Multi-vintage VMI backtest — same engine rules as simulate2 (2000 run):

  $1M/account, 16 stocks, 6.25% cap, 3 tranches; tranche 1 first week
  price < IV; adds only at 40w SMA x 1.01 AND under IV, >=8 weeks apart;
  sells ONLY on fraud/scandal with same-week full redeploy;
  IV = price_anchor x dcf_factor(g, beta) x ocf_mult / trailing_PE,
  growing at min(g, 10%)/yr; dcf_factor = 20y course DCF, r = RF + beta*MRP.

Vintages: 2000 (books from simulate2, extended to 2026), 2015, 2020
(books from the point-in-time scanner + manual era wide-moat pass with
anti-bubble slices — see vintage_books.py). Era RF per vintage.

Scandal-sell events applied consistently with the 2000 run's standard
(accounting fraud / criminal probe of core business, not product setbacks):
  * 2000A: BMY 2002-07 channel-stuffing -> PFE; UNH 2006-10 backdating -> KO
  * 2000B: CAH 2004-07 SEC accounting probe -> GPC
  * 2020 growth: UNH 2025-05 DOJ criminal probe (Medicare billing) ->
    next-ranked era wide-moat (computed, no hindsight performance peeking).
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
INITIAL = 1_000_000.0
N = 16
CAP = INITIAL / N
TRANCHE = CAP / 3.0
GAP = 8
MRP = 0.04
END = "2026-07-18"

# ---- 2000 books (verbatim from simulate2.py) ----
B2000_DEF = {
    "JNJ": (29, .125, .85, 1.15), "ABT": (21, .12, .80, 1.20),
    "BMY": (28, .12, .85, 1.10), "BDX": (19, .11, .75, 1.35),
    "UNH": (16, .15, .90, 1.30), "MRK": (28, .12, .90, 1.10),
    "PG":  (38, .11, .75, 1.25), "KMB": (20, .10, .80, 1.30),
    "CLX": (30, .11, .70, 1.25), "GIS": (17, .09, .65, 1.20),
    "HSY": (22, .10, .70, 1.20), "HRL": (16, .10, .65, 1.25),
    "PEP": (26, .11, .80, 1.30), "CL":  (33, .12, .80, 1.25),
    "MCD": (27, .11, .85, 1.50), "GD":  (12, .10, .75, 1.20),
}
B2000_GRO = {
    "TJX": (14, .15, .90, 1.30), "ROST": (12, .15, .95, 1.25),
    "AZO": (11, .14, .90, 1.30), "ORLY": (22, .22, .95, 1.20),
    "NKE": (22, .14, 1.00, 1.20), "SYY": (27, .14, .75, 1.25),
    "CAH": (22, .20, .90, 1.10), "ITW": (19, .14, .95, 1.20),
    "DHR": (24, .17, .95, 1.30), "LOW": (21, .22, 1.05, 1.20),
    "CVS": (23, .15, .85, 1.20), "TGT": (22, .15, .95, 1.40),
    "DLTR": (28, .25, 1.00, 1.20), "CHD": (18, .12, .70, 1.25),
    "SBUX": (42, .27, 1.10, 1.60), "SYK": (32, .20, .90, 1.30),
}


def load_book(year, key):
    d = json.load(open(os.path.join(HERE, f"books_{year}.json")))
    return {t: (v["pe"], v["g"], v["beta"], v["ocf_mult"])
            for t, v in d[key].items()}, d["rf"]


def dcf_factor(g, beta, rf):
    r = rf + beta * MRP
    f = sum((1 + g) ** t / (1 + r) ** t for t in range(1, 11))
    y10 = (1 + g) ** 10
    f += sum(y10 * 1.04 ** k / (1 + r) ** (10 + k) for k in range(1, 11))
    return f


def run_account(px, sma40, dates, book, repls, rf, label):
    anchor = dates[0]
    iv0, g_iv = {}, {}
    for t, (pe, g, b, m) in book.items():
        p = px.at[anchor, t]
        if np.isnan(p):
            p = px[t].loc[anchor:].dropna().iloc[0]
        iv0[t] = p * dcf_factor(g, b, rf) * m / pe
        g_iv[t] = min(g, 0.10)

    def iv_at(t, d):
        return iv0[t] * (1 + g_iv[t]) ** ((d - anchor).days / 365.25)

    sh, tr, last = {}, {}, {}
    cash = INITIAL
    slog, eq = [], []
    repl = {pd.Timestamp(d): (o, n) for d, o, n in repls}
    for i, d in enumerate(dates):
        for rd, (old, new) in list(repl.items()):
            if d >= rd and old in sh:
                proceeds = sh.pop(old) * px.at[d, old]
                tr.pop(old); last.pop(old)
                sh[new] = proceeds / px.at[d, new]
                tr[new] = 3; last[new] = i
                # replacement inherits an IV anchored at its entry
                bk = book.get(new)
                if bk is None:  # replacement not in original book: use old's params
                    bk = book[old]
                pe_, g_, b_, m_ = bk
                iv0[new] = px.at[d, new] * dcf_factor(g_, b_, rf) * m_ / pe_ / \
                    (1 + min(g_, .10)) ** ((d - anchor).days / 365.25)
                g_iv[new] = min(g_, 0.10)
                slog.append({"date": str(d.date()), "ticker": old, "action": "SELL",
                             "price": round(float(px.at[d, old]), 2),
                             "amount": round(proceeds)})
                slog.append({"date": str(d.date()), "ticker": new, "action": "BUY_REPL",
                             "price": round(float(px.at[d, new]), 2),
                             "amount": round(proceeds)})
                del repl[rd]
        for t in book:
            if t not in px.columns:
                continue
            p = px.at[d, t]
            if np.isnan(p):
                continue
            if t not in tr:
                if p < iv_at(t, d) and cash >= TRANCHE:
                    sh[t] = TRANCHE / p; tr[t] = 1; last[t] = i
                    cash -= TRANCHE
                    slog.append({"date": str(d.date()), "ticker": t, "action": "BUY",
                                 "tranche": 1, "price": round(float(p), 2),
                                 "amount": round(TRANCHE),
                                 "pct_of_iv": round(float(p / iv_at(t, d) * 100))})
            elif tr[t] < 3 and i - last[t] >= GAP:
                s = sma40.at[d, t]
                if not np.isnan(s) and p <= s * 1.01 and p < iv_at(t, d) \
                        and cash >= TRANCHE:
                    sh[t] += TRANCHE / p; tr[t] += 1; last[t] = i
                    cash -= TRANCHE
                    slog.append({"date": str(d.date()), "ticker": t, "action": "ADD",
                                 "tranche": tr[t], "price": round(float(p), 2),
                                 "amount": round(TRANCHE),
                                 "pct_of_iv": round(float(p / iv_at(t, d) * 100))})
        eq.append(cash + sum(s_ * px.at[d, t] for t, s_ in sh.items()
                             if not np.isnan(px.at[d, t])))
    ser = pd.Series(eq, index=dates, name=label)
    final = {t: round(s_ * px[t].dropna().iloc[-1]) for t, s_ in sh.items()}
    return ser, slog, final, round(cash)


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


def stats(ser, label):
    yrs = (ser.index[-1] - ser.index[0]).days / 365.25
    dd = drawdowns(ser)
    mx = max((x[3] for x in dd), default=0.0)
    return {"label": label,
            "final": round(float(ser.iloc[-1])),
            "total_return_pct": round((ser.iloc[-1] / INITIAL - 1) * 100, 1),
            "cagr_pct": round(((ser.iloc[-1] / INITIAL) ** (1 / yrs) - 1) * 100, 2),
            "years": round(yrs, 2),
            "n_corrections_7pct": len(dd),
            "max_drawdown_pct": round(mx * 100, 1)}


def main():
    out = {"vintages": {}}
    curves = {}

    # ---------- 2000 vintage (extended to 2026) ----------
    px00 = pd.read_csv(os.path.join(HERE, "weekly_adj_2026.csv"),
                       index_col=0, parse_dates=True).loc["1999-01-01":END]
    sma00 = px00.rolling(40).mean()
    dates00 = px00.loc["2000-01-01":END].index
    rf00 = 0.065
    eqA, logA, holdA, cashA = run_account(
        px00, sma00, dates00, B2000_DEF,
        [("2002-07-08", "BMY", "PFE"), ("2006-10-16", "UNH", "KO")],
        rf00, "2000 defensive")
    eqB, logB, holdB, cashB = run_account(
        px00, sma00, dates00, B2000_GRO,
        [("2004-07-12", "CAH", "GPC")], rf00, "2000 growth")
    # replacement params for 2000 (PFE/KO/GPC inherit old params in engine)
    spy00 = px00.loc[dates00, "SPY"]; spy00 = spy00 / spy00.iloc[0] * INITIAL
    out["vintages"]["2000"] = {
        "rf": rf00, "defensive": stats(eqA, "2000 defensive"),
        "growth": stats(eqB, "2000 growth"), "spy": stats(spy00, "2000 spy"),
        "final_holdings": {"defensive": holdA, "growth": holdB},
        "cash": {"defensive": cashA, "growth": cashB}}
    curves["2000_def"], curves["2000_gro"], curves["2000_spy"] = eqA, eqB, spy00
    json.dump({"defensive": logA, "growth": logB},
              open(os.path.join(HERE, "trades_2000ext.json"), "w"), indent=1)

    # ---------- 2015 / 2020 vintages ----------
    pxv = pd.read_csv(os.path.join(HERE, "weekly_vintage.csv"),
                      index_col=0, parse_dates=True).loc[:END]
    smav = pxv.rolling(40).mean()
    REPLS = {(2020, "growth_book"): [("2025-05-19", "UNH", "SYK")],
             (2015, "growth_book"): [], (2015, "defensive_book"): [],
             (2020, "defensive_book"): []}
    # SYK: next-ranked 2020 era wide-moat growth candidate (PE<=50) not
    # already in either 2020 book — chosen by era rank, not performance.
    for year in (2015, 2020):
        start = f"{year}-01-01"
        dates = pxv.loc[start:END].index
        vres = {}
        for key, lab in (("growth_book", "growth"), ("defensive_book", "defensive")):
            book, rf = load_book(year, key)
            # add replacement tickers' params if needed
            for _, old, new in REPLS.get((year, key), []):
                if new not in book:
                    book[new] = book[old]  # engine anchors new IV at entry
            book = {t.replace(".", "-"): v for t, v in book.items()}
            ser, slog, hold, cash = run_account(
                pxv, smav, dates, book, REPLS.get((year, key), []),
                rf, f"{year} {lab}")
            vres[lab] = stats(ser, f"{year} {lab}")
            vres.setdefault("final_holdings", {})[lab] = hold
            vres.setdefault("cash", {})[lab] = cash
            curves[f"{year}_{lab[:3]}"] = ser
            json.dump(slog, open(os.path.join(
                HERE, f"trades_{year}_{lab}.json"), "w"), indent=1)
        spy = pxv.loc[dates, "SPY"]; spy = spy / spy.iloc[0] * INITIAL
        vres["spy"] = stats(spy, f"{year} spy")
        vres["rf"] = load_book(year, "growth_book")[1]
        curves[f"{year}_spy"] = spy
        out["vintages"][str(year)] = vres

    pd.DataFrame(curves).to_csv(os.path.join(HERE, "vintage_curves.csv"))
    json.dump(out, open(os.path.join(HERE, "vintage_results.json"), "w"), indent=1)

    print(f"{'vintage':8} {'account':10} {'years':>5} {'final':>14} "
          f"{'total':>9} {'CAGR':>7} {'maxDD':>6} {'corr':>4}")
    for v, r in out["vintages"].items():
        for k in ("defensive", "growth", "spy"):
            s = r[k]
            print(f"{v:8} {k:10} {s['years']:5.1f} ${s['final']:>13,} "
                  f"{s['total_return_pct']:8.1f}% {s['cagr_pct']:6.2f}% "
                  f"{s['max_drawdown_pct']:5.1f}% {s['n_corrections_7pct']:4}")


if __name__ == "__main__":
    main()
