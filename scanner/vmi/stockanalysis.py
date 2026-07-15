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
