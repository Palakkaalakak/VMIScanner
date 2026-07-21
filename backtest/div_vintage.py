"""7-vintage all-growth backtest — NO scandal sells, dividends reinvested
with discipline, plus ideal-leverage recalculation.

Changes vs deep_vintage.py:
  * NO sells of any kind — scandal names are simply held through.
  * Explicit dividends: prices are split-adjusted but dividend-UNadjusted
    (weekly_unadj.csv, auto_adjust=False) with per-share weekly dividends
    (weekly_divs.csv). Dividend cash accrues to a separate pool and is
    reinvested ONLY with discipline: a book stock qualifies when it is
    on support (price <= 40w SMA x 1.01) AND under intrinsic value; the
    dividend pool buys the cheapest-vs-IV qualifying stock each week
    (needn't be the payer). Otherwise the cash waits.
  * Initial $1M deploys via the same 3-tranche program as before.
  * Ideal leverage: from each vintage's realized weekly returns, grid
    search L in [0, 5] maximizing CAGR of  1 + L*r_week - (L-1)*rf_week
    (borrow at era DGS10 path, weekly). Also report full-Kelly
    f* = mean(weekly excess) / var(weekly) — both purely data-derived.

Benchmark: dividend-adjusted store (weekly_deep.csv = total-return-ish
auto_adjust series): SPY where it exists at vintage start, else ^GSPC
(price-only — noted).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from multi_vintage import B2000_GRO, INITIAL, dcf_factor, stats  # noqa

N = 16
CAP = INITIAL / N
TRANCHE = CAP / 3.0
GAP = 8
END = "2026-07-18"

VINTAGES = {
    1990: "1990-01-05", 1995: "1995-01-06", 2000: "2000-01-07",
    2005: "2005-01-07", 2010: "2010-01-08", 2015: "2015-01-09",
    2020: "2020-01-10",
}
RF = {1990: .0794, 1995: .0788, 2000: .065, 2005: .0423,
      2010: .0385, 2015: .0212, 2020: .0188}


def load_book(year):
    if year == 2000:
        return {t: v for t, v in B2000_GRO.items()}
    d = json.load(open(os.path.join(HERE, f"books_growth_{year}.json")))
    return {t.replace(".", "-"): (v["pe"], v["g"], v["beta"], v["ocf_mult"])
            for t, v in d["growth_book"].items()}


def run(px, dv, sma, dates, book, rf, label):
    anchor = dates[0]
    iv0, g_iv = {}, {}
    for t, (pe, g, b, m) in book.items():
        s = px[t].loc[anchor:].dropna()
        if s.empty:
            continue
        iv0[t] = s.iloc[0] * dcf_factor(g, b, rf) * m / pe
        g_iv[t] = min(g, 0.10)

    def iv_at(t, d):
        return iv0[t] * (1 + g_iv[t]) ** ((d - anchor).days / 365.25)

    sh = {}          # shares
    tr, last = {}, {}
    cash = INITIAL   # tranche program pool
    div_cash = 0.0   # dividend pool
    slog, eq, div_total = [], [], 0.0
    for i, d in enumerate(dates):
        # 1) collect dividends
        for t, s_ in sh.items():
            dps = dv.at[d, t] if t in dv.columns else 0.0
            if dps and not np.isnan(dps):
                div_cash += s_ * dps
                div_total += s_ * dps
        # 2) tranche program (unchanged rules)
        for t in book:
            if t not in iv0 or t not in px.columns:
                continue
            p = px.at[d, t]
            if np.isnan(p):
                continue
            if t not in tr:
                if p < iv_at(t, d) and cash >= TRANCHE:
                    sh[t] = sh.get(t, 0) + TRANCHE / p
                    tr[t] = 1; last[t] = i; cash -= TRANCHE
                    slog.append({"date": str(d.date()), "ticker": t,
                                 "action": "BUY", "tranche": 1,
                                 "price": round(float(p), 2),
                                 "amount": round(TRANCHE)})
            elif tr[t] < 3 and i - last[t] >= GAP:
                s = sma.at[d, t]
                if not np.isnan(s) and p <= s * 1.01 and p < iv_at(t, d) \
                        and cash >= TRANCHE:
                    sh[t] += TRANCHE / p
                    tr[t] += 1; last[t] = i; cash -= TRANCHE
                    slog.append({"date": str(d.date()), "ticker": t,
                                 "action": "ADD", "tranche": tr[t],
                                 "price": round(float(p), 2),
                                 "amount": round(TRANCHE)})
        # 3) disciplined dividend reinvestment: cheapest-vs-IV stock that
        #    is on support AND under IV gets the whole pool this week
        if div_cash > 0:
            cands = []
            for t in book:
                if t not in iv0 or t not in px.columns:
                    continue
                p = px.at[d, t]
                s = sma.at[d, t]
                if np.isnan(p) or np.isnan(s):
                    continue
                ivd = iv_at(t, d)
                if p <= s * 1.01 and p < ivd:
                    cands.append((p / ivd, t, p))
            if cands:
                cands.sort()
                _, t, p = cands[0]
                sh[t] = sh.get(t, 0) + div_cash / p
                slog.append({"date": str(d.date()), "ticker": t,
                             "action": "DIV_REINVEST",
                             "price": round(float(p), 2),
                             "amount": round(div_cash)})
                div_cash = 0.0
        eq.append(cash + div_cash +
                  sum(s_ * px.at[d, t] for t, s_ in sh.items()
                      if not np.isnan(px.at[d, t])))
    ser = pd.Series(eq, index=dates, name=label)
    final = {t: round(s_ * px[t].dropna().iloc[-1]) for t, s_ in sh.items()}
    return ser, slog, final, round(cash + div_cash), round(div_total)


def ideal_leverage(ser, dgs10):
    """Grid-search L maximizing CAGR of leveraged weekly returns with
    borrowing at the DGS10 path; also full-Kelly estimate."""
    r = ser.pct_change(fill_method=None).dropna()
    rf_w = (dgs10.reindex(r.index, method="ffill") / 100 / 52).fillna(0)
    ex = r - rf_w
    kelly = float(ex.mean() / r.var())
    best_L, best_cagr = 1.0, None
    grid = np.arange(0.0, 5.01, 0.05)
    yrs = (ser.index[-1] - ser.index[0]).days / 365.25
    out = []
    for L in grid:
        lr = 1 + L * r.values - (L - 1) * rf_w.values
        if (lr <= 0).any():          # wiped out -> leverage infeasible
            out.append((round(L, 2), None))
            continue
        cagr = float(np.exp(np.log(lr).sum() / yrs) - 1)
        out.append((round(L, 2), round(cagr * 100, 2)))
        if best_cagr is None or cagr > best_cagr:
            best_cagr, best_L = cagr, L
    return {"kelly_full": round(kelly, 2),
            "kelly_half": round(kelly / 2, 2),
            "ideal_L_cagr_max": round(best_L, 2),
            "cagr_at_ideal_pct": round(best_cagr * 100, 2),
            "cagr_unlevered_pct": round(
                (float(ser.iloc[-1] / ser.iloc[0]) ** (1 / yrs) - 1) * 100, 2),
            "grid": out}


def main():
    px = pd.read_csv(os.path.join(HERE, "weekly_unadj.csv"),
                     index_col=0, parse_dates=True) \
        .resample("W-FRI").last().loc[:END]
    dv = pd.read_csv(os.path.join(HERE, "weekly_divs.csv"),
                     index_col=0, parse_dates=True) \
        .resample("W-FRI").sum().loc[:END]
    bench_adj = pd.read_csv(os.path.join(HERE, "weekly_deep.csv"),
                            index_col=0, parse_dates=True) \
        .resample("W-FRI").last().loc[:END]
    sma = px.rolling(40).mean()
    dgs10 = pd.read_csv(os.path.join(HERE, "dgs10.csv"),
                        index_col=0, parse_dates=True).iloc[:, 0]

    out, curves = {}, {}
    for year, start in VINTAGES.items():
        book = load_book(year)
        dates = px.loc[start:END].index
        ser, slog, hold, cash, div_total = run(
            px, dv, sma, dates, book, RF[year], f"{year}")
        bcol = "SPY"
        bs = bench_adj[bcol].loc[dates[0]:].dropna() \
            if bcol in bench_adj.columns else pd.Series(dtype=float)
        if bs.empty or bs.index[0] > dates[5]:
            bcol = "^GSPC"
            bs = bench_adj[bcol].loc[dates[0]:].dropna()
        bser = bs / bs.iloc[0] * INITIAL
        lev = ideal_leverage(ser, dgs10)
        out[str(year)] = {
            "rf": RF[year], "growth": stats(ser, f"{year}"),
            "benchmark": stats(bser, f"{year} {bcol}"),
            "bench_ticker": bcol, "cash_end": cash,
            "dividends_collected": div_total,
            "n_div_reinvest": sum(1 for x in slog
                                  if x["action"] == "DIV_REINVEST"),
            "leverage": {k: v for k, v in lev.items() if k != "grid"},
            "final_holdings": hold}
        curves[f"{year}_gro"] = ser
        curves[f"{year}_bench"] = bser
        json.dump(slog, open(os.path.join(
            HERE, f"trades_d{year}.json"), "w"), indent=1)
        json.dump(lev, open(os.path.join(
            HERE, f"leverage_{year}.json"), "w"), indent=1)

    pd.DataFrame(curves).to_csv(os.path.join(HERE, "div_curves.csv"))
    json.dump(out, open(os.path.join(HERE, "div_results.json"), "w"),
              indent=1)

    print(f"{'vint':5} {'yrs':>5} {'final':>15} {'CAGR':>7} {'maxDD':>6} "
          f"{'divs':>12} {'reinv':>5} | {'bench':>7} | "
          f"{'idealL':>6} {'CAGR@L':>7} {'Kelly':>5}")
    for y, r in out.items():
        s, b, L = r["growth"], r["benchmark"], r["leverage"]
        print(f"{y:5} {s['years']:5.1f} ${s['final']:>14,} "
              f"{s['cagr_pct']:6.2f}% {s['max_drawdown_pct']:5.1f}% "
              f"${r['dividends_collected']:>11,} {r['n_div_reinvest']:5} | "
              f"{b['cagr_pct']:6.2f}% | {L['ideal_L_cagr_max']:6.2f} "
              f"{L['cagr_at_ideal_pct']:6.2f}% {L['kelly_full']:5.2f}")


if __name__ == "__main__":
    main()
