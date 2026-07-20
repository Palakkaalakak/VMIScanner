"""Derive era DCF inputs for vintage-scan greats — all from point-in-time data.

For each GREAT business found by the PIT scan:
  trailing_PE = (vintage price x weighted-avg shares) / latest-FY net income
                (all point-in-time; shares from SEC facts, price from yfinance)
  g           = median of rev_cagr_5y / ni_cagr_5y from the truncated scan
                metrics (era-knowable growth, no invented caps)
  beta        = 5y weekly regression vs S&P 500 ending at the vintage date
  ocf_mult    = mean(CFO / NetIncome) over last 5 PIT fiscal years
  RF          = 10y US Treasury (FRED DGS10) on the vintage date; MRP 4%
                (same MRP convention as the 2000-vintage run)

Output: backtest/vintage_inputs_<YEAR>.json — candidate table for the
manual wide-moat selection pass (moat is user/manual per VMI).
"""
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scanner"))
from vmi import sec  # noqa: E402

sec.MAX_YEARS = 40
HERE = os.path.dirname(os.path.abspath(__file__))
VINTAGE_DATES = {2020: "2020-01-02", 2015: "2015-01-02", 2010: "2010-01-04"}


def fred_dgs10(date):
    """10y Treasury yield (decimal) on/just before `date` from FRED."""
    cache = os.path.join(HERE, "dgs10.csv")
    if not os.path.exists(cache):
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
        urllib.request.urlretrieve(url, cache)
    df = pd.read_csv(cache)
    df.columns = ["date", "val"]
    df = df[df["val"] != "."]
    df = df[df["date"] <= date]
    return float(df["val"].iloc[-1]) / 100.0


def shares_series(ticker):
    """end-date -> weighted-avg diluted/basic shares from SEC facts."""
    cik = sec._ticker_map().get(ticker.upper())
    if cik is None:
        return {}
    from vmi.http import get
    raw = get(sec.FACTS_URL.format(cik=cik), domain_hint="sec",
              cache_max_age=86400 * 7)
    facts = json.loads(raw).get("facts", {})
    out = {}
    for tag in ("WeightedAverageNumberOfDilutedSharesOutstanding",
                "WeightedAverageNumberOfSharesOutstandingBasic",
                "CommonStockSharesOutstanding"):
        f = facts.get("us-gaap", {}).get(tag) or facts.get("dei", {}).get(tag)
        if not f:
            continue
        for unit, es in f.get("units", {}).items():
            for e in es:
                if e.get("form") in sec.FORMS and e.get("end") and \
                        isinstance(e.get("val"), (int, float)):
                    out.setdefault(e["end"], float(e["val"]))
        if out:
            break
    return out


def build(year):
    vd = VINTAGE_DATES[year]
    scan = json.load(open(os.path.join(HERE, f"vintage_scan_{year}.json")))
    greats = [r for r in scan["results"] if r.get("is_great")]
    print(f"[{year}] {len(greats)} greats; deriving inputs...", flush=True)

    tickers = [g["ticker"].replace(".", "-") for g in greats]
    # 5y weekly prices ending at vintage for beta + vintage price
    px = yf.download(tickers + ["^GSPC"],
                     start=f"{year-5}-01-01", end=vd, interval="1wk",
                     auto_adjust=True, progress=False)["Close"]
    mkt = px["^GSPC"].pct_change(fill_method=None).dropna()
    rf = fred_dgs10(vd)

    out = []
    for g in greats:
        t = g["ticker"]
        yft = t.replace(".", "-")
        met = g.get("metrics", {}) or {}
        rev5 = met.get("rev_cagr_5y")
        ni5 = met.get("ni_cagr_5y")
        # g = sales growth first (VMI check #1; durable, no base effects),
        # NI growth only as fallback. No caps.
        growth = rev5 if rev5 is not None else ni5

        # beta
        beta = None
        if yft in px.columns:
            r = px[yft].pct_change(fill_method=None).dropna()
            j = r.index.intersection(mkt.index)
            if len(j) >= 100:
                rr, mm = r.loc[j], mkt.loc[j]
                beta = float(np.cov(rr, mm)[0, 1] / np.var(mm))

        # PIT fundamentals: latest FY NI, CFO/NI 5y, shares, PE
        pe = ocf_mult = None
        ni_latest = None
        try:
            data = sec.fetch_all(yft)
            inc, cfs = data.get("income", {}), data.get("cashflow", {})
            fy = inc.get("fiscalYear") or []
            keep = [i for i, f in enumerate(fy) if str(f) <= vd]
            ni = [inc.get("netIncome", [None] * len(fy))[i] for i in keep]
            cfo = [(cfs.get("ncfo") or [None] * len(fy))[i] for i in keep
                   if i < len(cfs.get("ncfo") or [])]
            pairs = [(c, n) for c, n in zip(cfo, ni) if c and n and n > 0][:5]
            if pairs:
                ocf_mult = float(np.mean([c / n for c, n in pairs]))
            ni_latest = next((v for v in ni if v), None)
            fy_dates = [fy[i] for i in keep]
            sh = shares_series(yft)
            sh_pit = next((sh.get(d) for d in fy_dates if sh.get(d)), None)
            if ni_latest and ni_latest > 0 and sh_pit and yft in px.columns:
                p0 = px[yft].dropna()
                if len(p0):
                    # yfinance Close is SPLIT-adjusted even with
                    # auto_adjust=False; SEC as-reported shares are
                    # pre-split. Actual era price = close x product of
                    # split ratios occurring AFTER the vintage date.
                    fac = 1.0
                    try:
                        spl = yf.Ticker(yft).splits
                        if spl is not None and len(spl):
                            after = spl[spl.index.tz_localize(None) > pd.Timestamp(vd)]
                            for ratio in after.values:
                                if ratio and ratio > 0:
                                    fac *= float(ratio)
                    except Exception:
                        pass
                    praw = yf.download(yft, start=vd, interval="1d",
                                       auto_adjust=False, progress=False,
                                       end=f"{year}-02-01")["Close"]
                    if len(praw):
                        price0 = float(praw.iloc[0].item() if hasattr(praw.iloc[0], "item") else praw.iloc[0])
                        pe = price0 * fac * sh_pit / ni_latest
        except Exception as e:
            print(f"   {t}: inputs error {e}", flush=True)

        out.append({
            "ticker": t, "sector": g.get("sector"), "industry": g.get("industry"),
            "score": g.get("score"), "fy_span": g.get("fy_span"),
            "rev_cagr_5y": rev5, "ni_cagr_5y": ni5, "growth": growth,
            "beta": None if beta is None else round(beta, 2),
            "trailing_pe": None if pe is None else round(pe, 1),
            "ocf_mult": None if ocf_mult is None else round(ocf_mult, 2),
            "roic_persistence": (g.get("moat_hints") or {}).get("roic_persistence", ""),
            "gross_margin": (g.get("moat_hints") or {}).get("gross_margin", ""),
            "operating_margin": (g.get("moat_hints") or {}).get("operating_margin", ""),
            "buybacks": (g.get("moat_hints") or {}).get("buybacks", ""),
        })

    res = {"vintage": year, "date": vd, "rf_10y": rf, "mrp": 0.04,
           "candidates": out}
    json.dump(res, open(os.path.join(HERE, f"vintage_inputs_{year}.json"), "w"),
              indent=1)
    ok = [c for c in out if c["trailing_pe"] and c["growth"] is not None
          and c["beta"] is not None and c["ocf_mult"]]
    print(f"[{year}] RF={rf*100:.2f}%  {len(ok)}/{len(out)} candidates with full inputs",
          flush=True)
    return res


if __name__ == "__main__":
    for y in [int(a) for a in sys.argv[1:]] or [2020]:
        build(y)
