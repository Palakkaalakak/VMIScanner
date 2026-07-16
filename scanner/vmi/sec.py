"""SEC Company Facts — primary long-history data source for US-filed issuers.

Why this exists: macrotrends.net throttles to ~1 request / 8s and 429s
concurrent sessions, making a full S&P500 pass take hours and ad-hoc
benchmark runs flaky. The SEC's official XBRL Company Facts API returns
EVERY filed annual fact for a company in ONE fast JSON request, with a
documented fair-access policy (10 req/s) instead of adversarial blocking.

Output shape mirrors macrotrends.py (the convention checks.py expects):
  {"income": {...}, "balance": {...}, "cashflow": {...}, "ratios": {...}}
where each section maps key -> newest-first list aligned on "fiscalYear",
and ratio fields are expressed as PERCENT (46.9 = 46.9%).

Caveats handled here:
  * Tag migration: companies move between US-GAAP tags over the years
    (e.g. SalesRevenueNet -> RevenueFromContractWithCustomerExcludingAssessedTax).
    Taking only the first matching tag truncates history, so each fiscal
    date is filled from a priority-ordered alias list instead.
  * DSGX and others use RevenueFromContractWithCustomerIncludingAssessedTax.
  * TMO reports plain "Depreciation" rather than the combined DD&A tag.
  * Duration facts must span a full year (>=250 days) to exclude
    quarterly/YTD observations that share the same annual end date.
  * Foreign issuers report in their home currency; pick the most-populated
    unit rather than assuming USD (ratios are unaffected by currency).
  * Stale filers: a ticker can stay in SEC's map after a foreign issuer
    stops filing in the US. If the latest annual anchor is older than ~2
    years, return None so scan.py falls through to Yahoo/macrotrends
    rather than scoring stale history as current.
"""
import json
import threading
import time
from typing import Dict, List, Optional

from .http import get

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
MAX_YEARS = 15

# Priority-ordered US-GAAP/IFRS tag aliases for each concept we need.
# (concept, is_duration) — duration facts need a start..end span filter.
TAGS = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet", "Revenues", "SalesRevenueGoodsNet",
        "Revenue", "RevenueFromContractsWithCustomers",
    ),
    "netIncome": ("NetIncomeLoss", "ProfitLoss",
                  "NetIncomeLossAvailableToCommonStockholdersBasic"),
    "operatingIncome": ("OperatingIncomeLoss", "ProfitLossFromOperatingActivities"),
    "pretaxIncome": (
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "ProfitLossBeforeTax",
    ),
    "incomeTax": ("IncomeTaxExpenseBenefit", "IncomeTaxExpenseContinuingOperations",
                  "IncomeTaxExpense"),
    "grossProfit": ("GrossProfit",),
    "costOfRevenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
                      "CostOfSales"),
    "depreciation": ("DepreciationDepletionAndAmortization", "Depreciation",
                     "DepreciationAmortizationAndAccretionNet",
                     "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment"),
    "interestExpense": ("InterestExpense", "InterestExpenseNonoperating",
                        "InterestIncomeExpenseNet", "InterestExpenseDebt"),
    "interestIncome": ("InvestmentIncomeInterest", "InterestIncomeOther"),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
             "CashAndCashEquivalents"),
    "shortTermInvestments": ("ShortTermInvestments", "MarketableSecuritiesCurrent",
                             "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
                             "AvailableForSaleSecuritiesCurrent",
                             "OtherShortTermInvestments"),
    "shortTermDebt": ("DebtCurrent", "LongTermDebtCurrent",
                      "ShortTermBorrowings", "LongTermDebtAndCapitalLeaseObligationsCurrent",
                      "CurrentPortionOfLongTermDebt"),
    "receivables": ("AccountsReceivableNetCurrent",
                    "AccountsNotesAndLoansReceivableNetCurrent",
                    "ReceivablesNetCurrent", "TradeAndOtherCurrentReceivables"),
    "equity": ("StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
               "Equity"),
    "longTermDebt": ("LongTermDebt", "LongTermDebtNoncurrent",
                     "LongTermDebtAndCapitalLeaseObligations",
                     "NoncurrentPortionOfNoncurrentBorrowings"),
    "currentAssets": ("AssetsCurrent", "CurrentAssets"),
    "currentLiabilities": ("LiabilitiesCurrent", "CurrentLiabilities"),
    "ncfo": ("NetCashProvidedByUsedInOperatingActivities",
             "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
             "CashFlowsFromUsedInOperatingActivities"),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets",
              "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"),
}
DURATION_KEYS = {"revenue", "netIncome", "operatingIncome", "pretaxIncome",
                 "incomeTax", "grossProfit", "costOfRevenue", "depreciation",
                 "interestExpense", "interestIncome", "ncfo", "capex"}

_ticker_map_cache: Optional[Dict[str, int]] = None
_ticker_map_lock = threading.Lock()  # parallel scan workers share this map


def _ticker_map() -> Dict[str, int]:
    """SEC ticker -> CIK map (single cached request, refreshed weekly)."""
    global _ticker_map_cache
    with _ticker_map_lock:
        if _ticker_map_cache is None:
            raw = get(TICKER_MAP_URL, domain_hint="sec", cache_max_age=86400 * 7)
            data = json.loads(raw)
            _ticker_map_cache = {row["ticker"].upper(): int(row["cik_str"])
                                 for row in data.values()}
    return _ticker_map_cache


def _annual_series(facts: Dict, aliases, duration: bool) -> Dict[str, float]:
    """Return end-date -> value, merging tag aliases in priority order.

    Companies migrate between tags over the years; taking only the first
    matching tag truncates history. Each date is filled from aliases in
    priority order (setdefault keeps the higher-priority tag's value).
    For each tag, prefer the most-populated unit — the issuer's actual
    reporting currency — instead of assuming USD.
    """
    merged: Dict[str, float] = {}
    for taxonomy in ("us-gaap", "ifrs-full"):
        tax = facts.get(taxonomy, {})
        for tag in aliases:
            fact = tax.get(tag)
            if not fact:
                continue
            units = fact.get("units", {})
            eligible = {u: es for u, es in units.items()
                        if "/shares" not in u and u not in ("shares", "pure")}
            if not eligible:
                eligible = units
            entries = max(eligible.values(), key=len, default=[])
            best_for_tag: Dict[str, tuple] = {}
            for e in entries:
                if (e.get("form") not in FORMS or not e.get("end")
                        or not isinstance(e.get("val"), (int, float))):
                    continue
                if duration:
                    start = e.get("start")
                    if not start:
                        continue
                    try:
                        span = (time.mktime(time.strptime(e["end"], "%Y-%m-%d"))
                                - time.mktime(time.strptime(start, "%Y-%m-%d")))
                        if span < 250 * 86400:  # exclude quarterly/YTD facts
                            continue
                    except ValueError:
                        continue
                key = e["end"]
                filed = e.get("filed", "")
                if key not in best_for_tag or filed >= best_for_tag[key][0]:
                    best_for_tag[key] = (filed, float(e["val"]))
            for date, (_filed, value) in best_for_tag.items():
                merged.setdefault(date, value)
    return merged


def _aligned(raw: Dict[str, Dict[str, float]], years: List[str], key: str
             ) -> List[Optional[float]]:
    d = raw.get(key, {})
    return [d.get(y) for y in years]


def _safe_div(a: Optional[float], b: Optional[float], scale: float = 1.0
              ) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b * scale


def fetch_all(ticker: str, use_cache: bool = True
              ) -> Optional[Dict[str, Dict[str, List]]]:
    """Fetch and reshape SEC Company Facts. None if not an SEC filer,
    if revenue history is absent, or if the filings are stale (delisted/
    deregistered foreign issuer still present in the ticker map)."""
    t = ticker.upper().replace(".", "-")
    cik = _ticker_map().get(t) or _ticker_map().get(t.replace("-", ""))
    if cik is None:
        return None
    try:
        raw_json = get(FACTS_URL.format(cik=cik), domain_hint="sec",
                       use_cache=use_cache)
    except LookupError:
        return None
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        return None
    facts = payload.get("facts", {})
    if not facts:
        return None

    raw = {key: _annual_series(facts, aliases, key in DURATION_KEYS)
           for key, aliases in TAGS.items()}

    anchor = raw["revenue"] or raw["netIncome"]
    if not anchor:
        return None
    years = sorted(anchor, reverse=True)[:MAX_YEARS]
    # Stale-filer guard: don't score old history as current fundamentals.
    try:
        latest_epoch = time.mktime(time.strptime(years[0], "%Y-%m-%d"))
        if time.time() - latest_epoch > 700 * 86400:
            return None
    except (ValueError, IndexError):
        return None

    revenue = _aligned(raw, years, "revenue")
    net_income = _aligned(raw, years, "netIncome")
    op_income = _aligned(raw, years, "operatingIncome")
    gross_profit = _aligned(raw, years, "grossProfit")
    cost_rev = _aligned(raw, years, "costOfRevenue")
    depreciation = _aligned(raw, years, "depreciation")
    equity = _aligned(raw, years, "equity")
    cur_assets = _aligned(raw, years, "currentAssets")
    cur_liab = _aligned(raw, years, "currentLiabilities")
    ncfo = _aligned(raw, years, "ncfo")
    capex = _aligned(raw, years, "capex")

    # Derive gross profit from cost of revenue when GrossProfit isn't tagged.
    gross_profit = [g if g is not None else
                    (r - c if r is not None and c is not None else None)
                    for g, r, c in zip(gross_profit, revenue, cost_rev)]

    income = {
        "fiscalYear": years,
        "revenue": revenue,
        "netIncome": net_income,
        "operatingIncome": op_income,
        "pretaxIncome": _aligned(raw, years, "pretaxIncome"),
        "incomeTax": _aligned(raw, years, "incomeTax"),
        "ebit": op_income,
        "ebitda": [(o + d) if o is not None and d is not None else None
                   for o, d in zip(op_income, depreciation)],
        "grossMargin": [_safe_div(g, r, 100) for g, r in zip(gross_profit, revenue)],
        "profitMargin": [_safe_div(n, r, 100) for n, r in zip(net_income, revenue)],
        "operatingMargin": [_safe_div(o, r, 100) for o, r in zip(op_income, revenue)],
        "income_statement_interest_expense": _aligned(raw, years, "interestExpense"),
        "interestIncome": _aligned(raw, years, "interestIncome"),
    }
    balance = {
        "fiscalYear": years,
        "equity": equity,
        "longTermDebt": _aligned(raw, years, "longTermDebt"),
        "cash": _aligned(raw, years, "cash"),
        "shortTermInvestments": _aligned(raw, years, "shortTermInvestments"),
        "shortTermDebt": _aligned(raw, years, "shortTermDebt"),
        "receivables": _aligned(raw, years, "receivables"),
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
    for ticker_arg in (sys.argv[1:] or ["AAPL"]):
        result = fetch_all(ticker_arg, use_cache=True)
        if result is None:
            print(ticker_arg, "-> no SEC data / stale filer")
        else:
            print(ticker_arg, "->", {
                name: len(section.get("fiscalYear", []))
                for name, section in result.items()})
