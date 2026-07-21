"""All-growth, sector-capped multi-vintage VMI backtest: 1990 -> 2020.

Engine rules identical to simulate2 / multi_vintage (imported, unchanged):
$1M, 16 stocks, 6.25% cap, 3 tranches, adds at 40w SMA under IV, sells
only on fraud/scandal with same-week redeploy. Era RF per vintage.

Vintages & price sources:
  1990, 1995, 2005, 2010 : weekly_deep.csv    (1985->2026, per-ticker fetch)
  2000                   : weekly_adj_2026.csv (dot-com book from simulate2 —
                           the deliberate sector-crash exception: wholly
                           anti-bubble, tech avoided on purpose)
  2015, 2020             : weekly_vintage.csv  (PIT scanner books, rebuilt
                           growth-only with 25% sector cap)

Benchmark: ^GSPC for deep vintages (SPY starts 1993), SPY thereafter.

Scandal sells (era-judged, same standard throughout):
  UNH Oct-2006 backdating      -> CL   (1995, 2005 books)
  WMT Apr-2012 FCPA Mexico     -> GIS  (1990, 1995, 2005, 2010 books)
  CAH Jul-2004 SEC acct probe  -> GPC  (2000 book)
  UNH May-2025 DOJ crim probe  -> CHD  (2010 book) / SYK (2020 book)
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from multi_vintage import (B2000_GRO, INITIAL, dcf_factor, drawdowns,  # noqa
                           run_account, stats)

END = "2026-07-18"

REPL_PARAMS = {  # era params for replacement tickers (PE, g, beta, ocf_mult)
    "CL":  (24, .10, .54, 1.30),   # at 2006-10 (beta computed)
    "GIS": (15, .08, .32, 1.20),   # at 2012-04
    "CHD": (24, .08, .24, 1.20),   # at 2025-05
}

VINTAGES = {
    1990: dict(src="deep", start="1990-01-02",
               repls=[("2012-04-23", "WMT", "GIS")]),
    1995: dict(src="deep", start="1995-01-03",
               repls=[("2006-10-16", "UNH", "CL"),
                      ("2012-04-23", "WMT", "GIS")]),
    2000: dict(src="2000", start="2000-01-03",
               repls=[("2004-07-12", "CAH", "GPC")]),
    2005: dict(src="deep", start="2005-01-03",
               repls=[("2006-10-16", "UNH", "CL"),
                      ("2012-04-23", "WMT", "GIS")]),
    2010: dict(src="deep", start="2010-01-04",
               repls=[("2012-04-23", "WMT", "GIS"),
                      ("2025-05-19", "UNH", "CHD")]),
    2015: dict(src="vint", start="2015-01-05", repls=[]),
    2020: dict(src="vint", start="2020-01-06",
               repls=[("2025-05-19", "UNH", "SYK")]),
}


def load_growth_book(year):
    d = json.load(open(os.path.join(HERE, f"books_growth_{year}.json")))
    return ({t: (v["pe"], v["g"], v["beta"], v["ocf_mult"])
             for t, v in d["growth_book"].items()}, d["rf"])


def main():
    px_deep = pd.read_csv(os.path.join(HERE, "weekly_deep.csv"),
                          index_col=0, parse_dates=True).loc[:END]
    px_2000 = pd.read_csv(os.path.join(HERE, "weekly_adj_2026.csv"),
                          index_col=0, parse_dates=True).loc[:END]
    px_vint = pd.read_csv(os.path.join(HERE, "weekly_vintage.csv"),
                          index_col=0, parse_dates=True).loc[:END]
    stores = {"deep": px_deep, "2000": px_2000, "vint": px_vint}
    smas = {k: v.rolling(40).mean() for k, v in stores.items()}

    out, curves = {}, {}
    for year, cfg in VINTAGES.items():
        px, sma = stores[cfg["src"]], smas[cfg["src"]]
        dates = px.loc[cfg["start"]:END].index
        if year == 2000:
            book, rf = dict(B2000_GRO), 0.065
        else:
            book, rf = load_growth_book(year)
        # add replacement params so new tickers get their own era inputs
        for _, old, new in cfg["repls"]:
            if new not in book and new in REPL_PARAMS:
                book[new] = REPL_PARAMS[new]
        book = {t.replace(".", "-"): v for t, v in book.items()
                if t.replace(".", "-") in px.columns}
        ser, slog, hold, cash = run_account(px, sma, dates, book,
                                            cfg["repls"], rf, f"{year}")
        bcol = "SPY"
        if "SPY" not in px.columns or px[bcol].loc[dates[0]:].dropna().empty \
                or px[bcol].loc[dates[0]:].dropna().index[0] > dates[5]:
            bcol = "^GSPC"
        bser = px.loc[dates, bcol].dropna()
        bser = bser / bser.iloc[0] * INITIAL
        st = stats(ser, f"{year} growth")
        sb = stats(bser, f"{year} {bcol}")
        out[str(year)] = {"rf": rf, "growth": st, "benchmark": sb,
                          "bench_ticker": bcol, "cash_end": cash,
                          "final_holdings": hold}
        curves[f"{year}_gro"] = ser
        curves[f"{year}_bench"] = bser
        json.dump(slog, open(os.path.join(
            HERE, f"trades_g{year}.json"), "w"), indent=1)

    pd.DataFrame(curves).to_csv(os.path.join(HERE, "deep_curves.csv"))
    json.dump(out, open(os.path.join(HERE, "deep_results.json"), "w"),
              indent=1)

    print(f"{'vintage':8} {'acct':7} {'yrs':>5} {'final':>15} {'total':>10} "
          f"{'CAGR':>7} {'maxDD':>6} {'corr':>4}")
    for y, r in out.items():
        for k, lab in (("growth", "growth"),
                       ("benchmark", r["bench_ticker"])):
            s = r[k]
            print(f"{y:8} {lab:7} {s['years']:5.1f} ${s['final']:>14,} "
                  f"{s['total_return_pct']:9.1f}% {s['cagr_pct']:6.2f}% "
                  f"{s['max_drawdown_pct']:5.1f}% {s['n_corrections_7pct']:4}")


if __name__ == "__main__":
    main()
