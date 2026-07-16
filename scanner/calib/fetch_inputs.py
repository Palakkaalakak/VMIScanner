"""Rebuild calibration inputs for the 11 StockOracle-screenshot benchmark
tickers: pulls SEC financials, finviz analyst estimates (incl. forward EPS
via Price/Forward-P/E), and stockanalysis.com's own forward EPS/growth
estimate table, then saves everything to calib_inputs.json / sa_est.json in
this directory so grid-search scripts can run repeatedly without re-fetching.

Run from repo root:  python3 -m scanner.calib.fetch_inputs
"""
import json
import os
import sys

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)

from scanner.vmi import sec, finviz  # noqa: E402
from scanner.vmi.http import get  # noqa: E402

TARGETS = {"AAPL": 251.12, "AMZN": 229.54, "GOOGL": 295.57, "MA": 571.00,
           "META": 906.66, "MSFT": 559.44, "NVDA": 221.02, "PANW": 202.00,
           "SPGI": 520.00, "TMO": 619.25, "WM": 230.98}
PRICES = {"AAPL": 327.50, "AMZN": 254.96, "GOOGL": 370.92, "MA": 535.21,
          "META": 681.31, "MSFT": 395.63, "NVDA": 212.50, "PANW": 354.02,
          "SPGI": 444.48, "TMO": 535.29, "WM": 232.80}


def _first(d, k):
    s = d.get(k) or []
    return next((v for v in s if v is not None), None)


def _deval(data):
    def resolve(i, depth=0):
        if depth > 14:
            return None
        v = data[i]
        if isinstance(v, dict):
            return {k: resolve(idx, depth + 1) for k, idx in v.items()}
        if isinstance(v, list):
            return [resolve(idx, depth + 1) for idx in v]
        return v
    return resolve(0)


def _find_key(o, key):
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def fetch_sa_estimates(ticker):
    """Pull stockanalysis.com's own forward-EPS + growth table for `ticker`."""
    raw = get(f"https://stockanalysis.com/stocks/{ticker.lower()}/forecast/__data.json",
              use_cache=True)
    payload = json.loads(raw)
    for node in payload.get("nodes", []):
        if not node or node.get("type") != "data":
            continue
        data = node.get("data")
        if not data or '"epsThis"' not in json.dumps(data):
            continue
        obj = _deval(data)
        ann = _find_key(obj, "annual")
        return {"annual": ann}
    return {"annual": None}


def main():
    tickers = list(TARGETS)
    est = finviz.fetch_growth_estimates(tickers, use_cache=True)
    out = {}
    sa_out = {}
    for t in tickers:
        m = sec.fetch_all(t, use_cache=True)
        inc, cf, bal = m["income"], m["cashflow"], m["balance"]
        e = est.get(t) or {}
        ni = _first(inc, "netIncome")
        sh = e.get("shares_outstanding")
        out[t] = {
            "ocf": _first(cf, "ncfo"), "fcf": _first(cf, "fcf"),
            "debt_lt": _first(bal, "longTermDebt") or 0,
            "debt_st": _first(bal, "shortTermDebt") or 0,
            "cash": _first(bal, "cash") or 0,
            "sti": _first(bal, "shortTermInvestments") or 0,
            "shares": sh, "beta": e.get("beta"), "g5": e.get("eps_next_5y"),
            "eps_next_y": e.get("eps_next_y"), "eps_this_y": e.get("eps_this_y"),
            "fwd_eps": e.get("fwd_eps"),
            "price": PRICES[t], "target_iv": TARGETS[t], "ni": ni,
            "eps": (ni / sh if ni and sh else None),
        }
        sa_out[t] = fetch_sa_estimates(t)
        print(t, "ocf/sh", round(out[t]["ocf"] / sh, 2) if out[t]["ocf"] and sh else None,
              "fwd_eps", out[t]["fwd_eps"], "beta", out[t]["beta"])
    json.dump(out, open(os.path.join(HERE, "calib_inputs.json"), "w"), indent=1)
    json.dump(sa_out, open(os.path.join(HERE, "sa_est.json"), "w"), indent=1)
    print("saved", len(out), "tickers")


if __name__ == "__main__":
    main()
