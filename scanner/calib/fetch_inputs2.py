"""Fetch SEC flows + finviz estimates for the EXPANDED 43-ticker benchmark
set (targets2.json: user's StockOracle screenshots with Base IV AND the
app's own Avg Growth Rates). Saves inputs2.json for calibration v2.

Run from repo root:  python3 scanner/calib/fetch_inputs2.py
"""
import json
import os
import sys

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)

from scanner.vmi import sec  # noqa: E402
from scanner.vmi.http import get  # noqa: E402
from scanner.vmi.finviz import _parse_est_rows, _EST_COLS, BASE  # noqa: E402

T = json.load(open(os.path.join(HERE, "targets2.json")))
tickers = [t for t in T if not t.startswith("_")]


def finviz_one(t):
    try:
        html = get(f"{BASE}?v=152&t={t}&c={_EST_COLS}", use_cache=True)
        rows = _parse_est_rows(html)
        return rows.get(t.upper().replace(".", "-")) or rows.get(t.upper())
    except Exception:
        return None


def _first(seq):
    return next((v for v in (seq or []) if v is not None), None)


out = {}
for t in tickers:
    print(t, end=" ", flush=True)
    d = sec.fetch_all(t)
    fv = finviz_one(t)
    rec = {"target_iv": T[t]["base_iv"], "so_growth": T[t]["growth"],
           "sector": T[t]["sector"], "finviz": fv}
    if d:
        cfd, bald, incd = d["cashflow"], d["balance"], d["income"]
        rec["sec"] = {
            "fy": incd["fiscalYear"][:8],
            "ocf": (cfd.get("ncfo") or [])[:8],
            "capex": (cfd.get("capex") or [])[:8],
            "netIncome": (incd.get("netIncome") or [])[:8],
            "revenue": (incd.get("revenue") or [])[:8],
            "cash": _first(bald.get("cash")),
            "sti": _first(bald.get("shortTermInvestments")),
            "std": _first(bald.get("shortTermDebt")),
            "ltd": _first(bald.get("longTermDebt")),
        }
    else:
        rec["sec"] = None
    out[t] = rec
print()
json.dump(out, open(os.path.join(HERE, "inputs2.json"), "w"), indent=1)
have_sec = sum(1 for r in out.values() if r["sec"])
have_fv = sum(1 for r in out.values() if r["finviz"])
print(f"saved inputs2.json: {len(out)} tickers, {have_sec} with SEC, {have_fv} with finviz")
