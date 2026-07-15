"""Yahoo Finance annual fundamentals — fallback for non-SEC/OTC issuers.

Some names in the user's benchmark list (CNSWF — Constellation Software,
EVVTY — Evolution AB) trade OTC in the US but do not file with the SEC,
and macrotrends' coverage for them is unreliable (429s / missing pages).
Yahoo's public fundamentals-timeseries endpoint exposes ~4-5 annual years
for these — not enough for the long consistency trends (those become NA),
but enough for the ratio checks (ROE, margins, current ratio, Debt/EBITDA,
debt servicing, receivables-vs-sales), which is sufficient applicable
coverage for an honest verdict.

Output shape mirrors macrotrends.py / sec.py:
  {"income": ..., "balance": ..., "cashflow": ..., "ratios": ...}
with newest-first lists aligned on "fiscalYear" and ratios in PERCENT.
"""
import datetime
import json
from typing import Dict, List, Optional

from .http import get

BASE = ("https://query2.finance.yahoo.com/ws/fundamentals-timeseries/"
        "v1/finance/timeseries/{symbol}")

_TYPES = [
    "annualTotalRevenue", "annualNetIncome", "annualOperatingIncome",
    "annualGrossProfit", "annualPretaxIncome", "annualTaxProvision",
    "annualEBIT", "annualInterestExpense",
    "annualOperatingCashFlow", "annualCapitalExpenditure",
    "annualStockholdersEquity", "annualLongTermDebt",
    "annualCashAndCashEquivalents", "annualAccountsReceivable",
    "annualCurrentAssets", "annualCurrentLiabilities",
]


def _fetch_raw(symbol: str, use_cache: bool = True) -> Optional[Dict[str, Dict[str, float]]]:
    """Return {concept: {asOfDate: value}} or None if Yahoo has nothing."""
    end = int(datetime.datetime.now().timestamp())
    url = (BASE.format(symbol=symbol)
           + f"?symbol={symbol}&type={','.join(_TYPES)}"
           + f"&period1=946684800&period2={end}")
    try:
        raw = get(url, domain_hint="yahoo", use_cache=use_cache)
        payload = json.loads(raw)
    except Exception:  # noqa: BLE001
        return None
    out: Dict[str, Dict[str, float]] = {}
    for row in payload.get("timeseries", {}).get("result", []):
        typ = (row.get("meta", {}).get("type") or [None])[0]
        if not typ:
            continue
        series = {}
        for point in (row.get(typ) or []):
            if not point:
                continue
            date = point.get("asOfDate")
            val = (point.get("reportedValue") or {}).get("raw")
            if date and isinstance(val, (int, float)):
                series[date] = float(val)
        if series:
            out[typ] = series
    return out or None


def _aligned(raw: Dict, years: List[str], typ: str) -> List[Optional[float]]:
    d = raw.get(typ, {})
    return [d.get(y) for y in years]


def _safe_div(a, b, scale: float = 1.0) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b * scale


def fetch_all(ticker: str, use_cache: bool = True
              ) -> Optional[Dict[str, Dict[str, List]]]:
    raw = _fetch_raw(ticker.upper(), use_cache=use_cache)
    if raw is None:
        return None
    anchor = raw.get("annualTotalRevenue") or raw.get("annualNetIncome")
    if not anchor:
        return None
    years = sorted(anchor, reverse=True)

    revenue = _aligned(raw, years, "annualTotalRevenue")
    net_income = _aligned(raw, years, "annualNetIncome")
    op_income = _aligned(raw, years, "annualOperatingIncome")
    gross_profit = _aligned(raw, years, "annualGrossProfit")
    ebit = _aligned(raw, years, "annualEBIT")
    equity = _aligned(raw, years, "annualStockholdersEquity")
    cur_assets = _aligned(raw, years, "annualCurrentAssets")
    cur_liab = _aligned(raw, years, "annualCurrentLiabilities")
    ncfo = _aligned(raw, years, "annualOperatingCashFlow")
    # Yahoo reports capex as a negative outflow; normalize to positive spend.
    capex = [abs(v) if v is not None else None
             for v in _aligned(raw, years, "annualCapitalExpenditure")]

    income = {
        "fiscalYear": years,
        "revenue": revenue,
        "netIncome": net_income,
        "operatingIncome": op_income,
        "pretaxIncome": _aligned(raw, years, "annualPretaxIncome"),
        "incomeTax": _aligned(raw, years, "annualTaxProvision"),
        "ebit": [e if e is not None else o for e, o in zip(ebit, op_income)],
        # Yahoo doesn't expose D&A on this endpoint; EBIT is the honest
        # conservative stand-in (overstates Debt/EBITDA slightly).
        "ebitda": [e if e is not None else o for e, o in zip(ebit, op_income)],
        "grossMargin": [_safe_div(g, r, 100) for g, r in zip(gross_profit, revenue)],
        "profitMargin": [_safe_div(n, r, 100) for n, r in zip(net_income, revenue)],
        "operatingMargin": [_safe_div(o, r, 100) for o, r in zip(op_income, revenue)],
        "income_statement_interest_expense": _aligned(raw, years, "annualInterestExpense"),
    }
    balance = {
        "fiscalYear": years,
        "equity": equity,
        "longTermDebt": _aligned(raw, years, "annualLongTermDebt"),
        "cash": _aligned(raw, years, "annualCashAndCashEquivalents"),
        "receivables": _aligned(raw, years, "annualAccountsReceivable"),
    }
    cashflow = {
        "fiscalYear": years,
        "ncfo": ncfo,
        "capex": capex,
        "fcf": [(o - (c or 0)) if o is not None else None
                for o, c in zip(ncfo, capex)],
    }
    ratios = {
        "fiscalYear": years,
        "roe": [_safe_div(n, e, 100) for n, e in zip(net_income, equity)],
        "currentRatio": [_safe_div(a, l) for a, l in zip(cur_assets, cur_liab)],
        "grossMargin": income["grossMargin"],
        "profitMargin": income["profitMargin"],
        "operatingMargin": income["operatingMargin"],
    }
    return {"income": income, "balance": balance, "cashflow": cashflow,
            "ratios": ratios}


if __name__ == "__main__":
    import sys
    for ticker_arg in (sys.argv[1:] or ["CNSWF"]):
        result = fetch_all(ticker_arg, use_cache=False)
        if result is None:
            print(ticker_arg, "-> no Yahoo data")
        else:
            print(ticker_arg, "->", {
                name: len(section.get("fiscalYear", []))
                for name, section in result.items()})
