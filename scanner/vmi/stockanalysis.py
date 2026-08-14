"""stockanalysis.com scraper — deep per-ticker financial history.

Fetches the SvelteKit __data.json payloads for income statement, balance
sheet, cash flow statement and ratios (annual, 10Y range) and decodes the
devalue-serialized node into plain {metric: [values...]} dicts.

Values are ordered newest-first; the first column may be TTM/Current
(fiscalYear like "TTM" or a future year); we keep everything and let the
checks layer decide.
"""
import json
from typing import Dict, List, Optional

from .http import get

STATEMENTS = {
    "income": "financials",
    "balance": "financials/balance-sheet",
    "cashflow": "financials/cash-flow-statement",
    "ratios": "financials/ratios",
}


def _decode_node(payload: dict) -> Optional[dict]:
    """Decode the last data node of a SvelteKit __data.json response."""
    nodes = [n for n in payload.get("nodes", []) if n and n.get("type") == "data"]
    if not nodes:
        return None
    data = nodes[-1]["data"]

    def deref(idx):
        if not isinstance(idx, int) or idx < 0 or idx >= len(data):
            return None
        v = data[idx]
        if isinstance(v, dict):
            return {k: deref(i) for k, i in v.items()}
        if isinstance(v, list):
            return [deref(i) for i in v]
        return v

    root = data[0]
    if not isinstance(root, dict):
        return None
    return {k: deref(i) for k, i in root.items()}


def fetch_statement(ticker: str, statement: str, use_cache: bool = True) -> Optional[Dict[str, List]]:
    """Return {'fiscalYear': [...], 'revenue': [...], ...} newest-first."""
    path = STATEMENTS[statement]
    url = f"https://stockanalysis.com/stocks/{ticker.lower()}/{path}/__data.json?p=10Y"
    try:
        raw = get(url, use_cache=use_cache)
    except LookupError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    root = _decode_node(payload)
    if not root:
        return None
    fd = root.get("financialData")
    if not isinstance(fd, dict):
        return None
    return fd


def fetch_profile(ticker: str, use_cache: bool = True) -> Dict[str, str]:
    """Return {'sector':..., 'industry':...} from the company profile page."""
    url = f"https://stockanalysis.com/stocks/{ticker.lower()}/company/__data.json"
    try:
        raw = get(url, use_cache=use_cache)
        root = _decode_node(json.loads(raw))
    except Exception:  # noqa: BLE001
        return {}
    prof = (root or {}).get("profile") or {}
    out = {}
    for k in ("sector", "industry"):
        v = prof.get(k)
        if isinstance(v, dict):
            v = v.get("value")
        if isinstance(v, str):
            out[k] = v
    return out


def fetch_all(ticker: str, use_cache: bool = True) -> Optional[Dict[str, Dict[str, List]]]:
    """Fetch all four statements. Returns None if income statement missing."""
    out = {}
    for name in STATEMENTS:
        fd = fetch_statement(ticker, name, use_cache=use_cache)
        if fd is None and name == "income":
            return None
        out[name] = fd or {}
    return out


def fetch_statistics(ticker: str, use_cache: bool = True) -> Dict[str, float]:
    """TTM statistics from the /statistics page (S&P Global Market
    Intelligence data — matches StockOracle's displayed TTM values: for MSFT
    ROE 34.04 exact, PE 27.68 vs 27.69, FCF yield 1.82 vs 1.81).

    Returns {} on failure. Keys: roe, roic, roa, pe, fwd_pe, peg, fcf_yield,
    div_yield, shares_out, current_ratio, quick_ratio, debt_equity,
    debt_ebitda, interest_coverage, z_score, f_score, eps_growth_3y,
    revenue_growth_3y, beta, gross_margin, operating_margin, profit_margin.
    """
    url = f"https://stockanalysis.com/stocks/{ticker.lower()}/statistics/__data.json"
    try:
        root = _decode_node(json.loads(get(url, use_cache=use_cache)))
    except Exception:  # noqa: BLE001
        return {}
    if not root:
        return {}
    # The page is a list of sections, each {data:[{id,title,value,hover},...]}
    want = {
        "roe": "roe", "roic": "roic", "roa": "roa",
        "pe": "pe", "peForward": "fwd_pe", "pegRatio": "peg",
        "fcfYield": "fcf_yield", "dividendYield": "div_yield",
        "sharesout": "shares_out",
        "currentRatio": "current_ratio", "quickRatio": "quick_ratio",
        "debtEquity": "debt_equity", "debtEbitda": "debt_ebitda",
        "interestCoverage": "interest_coverage",
        "zScore": "z_score", "fScore": "f_score",
        "eps3y": "eps_growth_3y", "revenue3y": "revenue_growth_3y",
        "beta": "beta", "grossMargin": "gross_margin",
        "operatingMargin": "operating_margin", "profitMargin": "profit_margin",
    }
    out: Dict[str, float] = {}

    def visit(o):
        if isinstance(o, dict):
            _id = o.get("id")
            if _id in want and "hover" in o:
                h = str(o["hover"]).replace(",", "").replace("%", "") \
                    .replace("$", "").strip()
                try:
                    out[want[_id]] = float(h)
                except ValueError:
                    pass
            for v in o.values():
                visit(v)
        elif isinstance(o, list):
            for v in o:
                visit(v)
    visit(root)
    return out
