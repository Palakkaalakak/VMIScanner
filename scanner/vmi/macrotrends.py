"""macrotrends.net scraper — up to 15 years of ANNUAL financial history.

Why this exists: stockanalysis.com (our original source) caps free-tier
history at ~5 fiscal years + TTM regardless of the `?p=` query param
(verified by direct testing). The VMI docs are only explicit about a 5y
window for ROE and margins; other metrics (ROIC, Sales/NI/CFO
"consistency") are described ambiguously as "5-10 years" or left
unspecified. To let `checks.py` actually apply a *real* 10-year window
where the docs don't pin 5y, we need a source with >5y of data.

macrotrends embeds all years of a statement in one inline JS array:
    var originalData = [{"field_name": "Revenue", "2025-09-30": "...",
                          "2024-09-30": "...", ...}, {...}, ...];
on four pages per ticker:
    /stocks/charts/<TICKER>/<any-slug>/income-statement
    /stocks/charts/<TICKER>/<any-slug>/balance-sheet
    /stocks/charts/<TICKER>/<any-slug>/cash-flow-statement
    /stocks/charts/<TICKER>/<any-slug>/financial-ratios
The <any-slug> segment is ignored by the server — macrotrends redirects
any placeholder slug to the ticker's canonical URL — so we can always
request with a dummy "x" slug and let `requests` follow the redirect.

Rate limiting: macrotrends aggressively 429s clients that hit it too
fast. `http.py` applies a much longer per-request throttle for this
domain (see _DOMAIN_MIN_INTERVAL["macrotrends"] there) — do not lower it
without re-testing; empirically ~8s between requests was reliably clean.
"""
import json
import re
from typing import Dict, List, Optional

from .http import get

STATEMENTS = {
    "income": "income-statement",
    "balance": "balance-sheet",
    "cashflow": "cash-flow-statement",
    "ratios": "financial-ratios",
}

# macrotrends field_name (post HTML-strip) -> our canonical key.
# Only fields actually used by checks.py are mapped; anything else in the
# page is ignored.
FIELD_MAP = {
    # income statement
    "Revenue": "revenue",
    "Cost Of Goods Sold": "cogs",
    "Gross Profit": "grossProfit",
    "Operating Income": "operatingIncome",
    "Net Income": "netIncome",
    "EBITDA": "ebitda",
    "EBIT": "ebit",
    "Pre-Tax Income": "pretaxIncome",
    "Income Taxes": "incomeTax",
    # balance sheet
    "Cash On Hand": "cash",
    "Receivables": "receivables",
    "Total Current Assets": "currentAssets",
    "Total Assets": "assets",
    "Total Current Liabilities": "currentLiabilities",
    "Long Term Debt": "longTermDebt",
    "Total Liabilities": "liabilities",
    "Share Holder Equity": "equity",
    # cash flow statement
    "Net Income/Loss": "cf_netIncome",
    "Cash Flow From Operating Activities": "ncfo",
    "Cash Flow From Investing Activities": "ncfi",
    "Net Change In Property, Plant, And Equipment": "capex",
    "Cash Flow From Financial Activities": "ncff",
    "Common Stock Dividends Paid": "dividendsPaid",
    "Net Common Equity Issued/Repurchased": "netEquityIssued",
    "Net Long-Term Debt": "netLongTermDebtIssued",
    "Net Current Debt": "netCurrentDebtIssued",
    # financial ratios
    "Current Ratio": "currentRatio",
    "Debt/Equity Ratio": "debtEquity",
    "Gross Margin": "grossMargin",
    "Operating Margin": "operatingMargin",
    "EBITDA Margin": "ebitdaMargin",
    "Net Profit Margin": "profitMargin",
    "ROE - Return On Equity": "roe",
    "ROA - Return On Assets": "roa",
    "ROI - Return On Investment": "roic",  # macrotrends' "ROI" ~= ROIC proxy
    "Days Sales In Receivables": "daysSalesReceivables",
}

# These macrotrends fields are already expressed as percent (e.g. "12.34"
# meaning 12.34%), unlike stockanalysis's fractional convention — no
# scaling needed here. Kept as an explicit set for clarity / future-proofing.
ALREADY_PERCENT = {
    "grossMargin", "operatingMargin", "ebitdaMargin", "profitMargin",
    "roe", "roa", "roic",
}


def _extract_original_data(html: str) -> List[Dict]:
    m = re.search(r"var\s+originalData\s*=\s*(\[.*?\])\s*;\s*\n", html, re.S)
    if not m:
        # fallback: some pages terminate the statement differently
        m = re.search(r"var\s+originalData\s*=\s*(\[.*\])\s*;", html, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return []


def _clean_field_name(raw: str) -> str:
    return re.sub(r"<[^>]+>", "", raw).strip()


def fetch_statement(ticker: str, statement: str, use_cache: bool = True,
                     max_years: int = 15) -> Optional[Dict[str, List]]:
    """Return {'fiscalYear': ['2025','2024',...], 'revenue': [...], ...}
    newest-first, for up to `max_years` of ANNUAL data.

    Returns None if the page has no usable data (e.g. ticker not covered).
    """
    path = STATEMENTS[statement]
    url = f"https://www.macrotrends.net/stocks/charts/{ticker.upper()}/x/{path}"
    try:
        html = get(url, use_cache=use_cache, domain_hint="macrotrends")
    except LookupError:
        return None
    rows = _extract_original_data(html)
    if not rows:
        return None

    # Collect the union of date columns across all rows, sorted desc.
    date_keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", k) and k not in date_keys:
                date_keys.append(k)
    date_keys.sort(reverse=True)
    date_keys = date_keys[:max_years]

    out: Dict[str, List] = {"fiscalYear": [d[:4] for d in date_keys]}
    for row in rows:
        name = _clean_field_name(row.get("field_name", ""))
        key = FIELD_MAP.get(name)
        if not key:
            continue
        series = []
        for d in date_keys:
            v = row.get(d)
            try:
                series.append(float(v) if v not in (None, "", "NA") else None)
            except (TypeError, ValueError):
                series.append(None)
        out[key] = series
    return out


def fetch_all(ticker: str, use_cache: bool = True) -> Optional[Dict[str, Dict[str, List]]]:
    """Fetch all four statements for a ticker. None if income stmt missing."""
    out = {}
    for name in STATEMENTS:
        fd = fetch_statement(ticker, name, use_cache=use_cache)
        if fd is None and name == "income":
            return None
        out[name] = fd or {}
    return out


if __name__ == "__main__":
    import sys
    import time
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    for stmt in STATEMENTS:
        fd = fetch_statement(t, stmt, use_cache=False)
        print(stmt, "->", None if fd is None else {k: v[:3] for k, v in list(fd.items())[:5]})
        time.sleep(8)
