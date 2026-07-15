"""VMI 'Great Business' checks (fundamentals only — no valuation, no TA).

Implements Adam Khoo's checklist from Lessons 4 & 7 / Quick Reference:

Profitability
  1. Sales revenue consistently increasing (5y)
  2. Net income (or operating income fallback) consistently increasing (5y)
  3. Cash Flow from Operations consistently increasing (5y)
  4. Free Cash Flow consistently positive
  5. Gross margin consistent or increasing (5y)
  6. Net margin consistent or increasing (5y)
  7. ROE >= 12% (target 12-15%+)
  8. ROIC >= 12% (n/a banks)

Financial strength
  9.  Current ratio >= 1
  10. Debt / EBITDA <= 3
  11. Debt servicing ratio (net interest expense / CFO) < 30%
  REITs: Gearing (Total Debt / Total Assets) < 45%
  Banks: CET1 > 10% & NPL < 5% (not available free -> flagged for manual check)

Management effectiveness
  12. Receivables growing no faster than sales (5y CAGR comparison)

Forward-looking
  13. Positive projected growth (enforced by the Finviz pre-filter:
      EPS next-5Y estimate positive)

Moat is NOT auto-decided (per user instruction): we compute *hints*
(margin level/stability, ROIC persistence, buybacks) and leave the final
moat call to user discretion.

Special-case handling per the course:
  - Banks / insurance / financial firms: skip current ratio, D/EBITDA,
    debt servicing, ROIC, CFO consistency; use net income consistency.
  - REITs: skip standard debt ratios; use gearing < 45%; CFO rules relaxed.
  - Property developers / commodity producers: CFO consistency n/a.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PASS, FAIL, WARN, NA = "PASS", "FAIL", "WARN", "NA"

# stockanalysis.com serves these as fractions (0.25 = 25%); convert to %.
PERCENT_KEYS = {
    "roe", "roic", "roa", "roce", "grossMargin", "profitMargin",
    "operatingMargin", "buybackyield", "payoutratio", "dividendyield",
    "fcfMargin", "ebitdaMargin", "ebitMargin", "earningsyield", "fcfyield",
}

# ---------------------------------------------------------------- helpers


def _series(fd: Dict, key: str, max_n: int = 6) -> List[Optional[float]]:
    """Newest-first numeric series for annual fiscal years (skip TTM col).

    stockanalysis returns col 0 as TTM/Current when it exists, then FY
    newest->oldest. We drop col 0 when it's a TTM/current column and keep
    up to `max_n` fiscal years.
    """
    vals = fd.get(key)
    years = fd.get("fiscalYear") or []
    if not vals:
        return []
    out = list(zip(years, vals))

    fq = fd.get("fiscalQuarter") or []
    # The first column is TTM/Current when its quarter label differs from
    # the modal (most common) quarter label of the rest — annual rows all
    # carry the same fiscal-year-end quarter label.
    if len(fq) >= 3 and fq[0] is not None:
        rest = [str(q) for q in fq[1:] if q is not None]
        if rest:
            modal = max(set(rest), key=rest.count)
            if str(fq[0]) != modal:
                out = out[1:]

    vals2 = [v for (_y, v) in out][:max_n]
    scale = 100.0 if key in PERCENT_KEYS else 1.0
    return [float(v) * scale if isinstance(v, (int, float)) else None for v in vals2]


def _oldest_first(s: List[Optional[float]]) -> List[float]:
    return [v for v in reversed(s) if v is not None]


def _consistently_increasing(s_newest_first: List[Optional[float]],
                             tolerance_dips: int = 1,
                             min_points: int = 4) -> Optional[bool]:
    """True if series trends up with at most `tolerance_dips` down-years
    AND the last value is above the first (net growth over the window)."""
    s = _oldest_first(s_newest_first)
    if len(s) < min_points:
        return None
    dips = sum(1 for a, b in zip(s, s[1:]) if b < a)
    return dips <= tolerance_dips and s[-1] > s[0]


def _consistent_or_increasing_margin(s_newest_first: List[Optional[float]],
                                     max_drop_pp_frac: float = 0.15,
                                     min_points: int = 4) -> Optional[bool]:
    """Margins must be consistent or increasing: allow small wobble.
    Fails if the latest margin is >15% (relative) below the 5y average
    or the series has collapsed vs its start."""
    s = _oldest_first(s_newest_first)
    if len(s) < min_points:
        return None
    avg = sum(s) / len(s)
    if avg <= 0:
        return False
    latest = s[-1]
    rel_drop_vs_avg = (avg - latest) / abs(avg)
    rel_drop_vs_start = (s[0] - latest) / abs(s[0]) if s[0] != 0 else 0
    return rel_drop_vs_avg <= max_drop_pp_frac and rel_drop_vs_start <= 0.25


def _cagr(s_newest_first: List[Optional[float]]) -> Optional[float]:
    s = _oldest_first(s_newest_first)
    if len(s) < 2 or s[0] is None or s[-1] is None or s[0] <= 0:
        return None
    n = len(s) - 1
    if s[-1] <= 0:
        return None
    return (s[-1] / s[0]) ** (1 / n) - 1


def _latest(fd: Dict, key: str) -> Optional[float]:
    s = _series(fd, key)
    for v in s:
        if v is not None:
            return v
    return None


def _avg(s: List[Optional[float]], n: int = 5) -> Optional[float]:
    vals = [v for v in s[:n] if v is not None]
    return sum(vals) / len(vals) if vals else None


# ---------------------------------------------------------------- classify

FINANCIAL_INDUSTRY_HINTS = (
    "bank", "insurance", "capital markets", "asset management",
    "financial data", "credit services", "mortgage", "financial conglomerates",
)
REIT_HINT = "reit"
PROPERTY_HINTS = ("real estate development", "real estate services",
                  "real estate - development")
COMMODITY_HINTS = ("oil", "gas", "coal", "gold", "silver", "copper", "steel",
                   "aluminum", "mining", "chemicals", "agricultural inputs")


def classify(sector: str, industry: str) -> str:
    s = (sector or "").lower()
    i = (industry or "").lower()
    if REIT_HINT in i:
        return "reit"
    if any(h in i for h in FINANCIAL_INDUSTRY_HINTS):
        return "financial"
    if s in ("financial", "financials"):
        return "financial"
    if any(h in i for h in PROPERTY_HINTS):
        return "property"
    if any(h in i for h in COMMODITY_HINTS):
        return "commodity"
    return "standard"


# ---------------------------------------------------------------- results

@dataclass
class CheckResult:
    name: str
    status: str          # PASS / FAIL / WARN / NA
    value: Optional[str] = None
    detail: str = ""


@dataclass
class ScanResult:
    ticker: str
    company: str = ""
    sector: str = ""
    industry: str = ""
    country: str = ""
    market_cap: str = ""
    company_type: str = "standard"
    checks: List[CheckResult] = field(default_factory=list)
    moat_hints: Dict[str, str] = field(default_factory=dict)
    error: str = ""

    @property
    def n_pass(self):
        return sum(1 for c in self.checks if c.status == PASS)

    @property
    def n_fail(self):
        return sum(1 for c in self.checks if c.status == FAIL)

    @property
    def n_warn(self):
        return sum(1 for c in self.checks if c.status == WARN)

    @property
    def applicable(self):
        return sum(1 for c in self.checks if c.status in (PASS, FAIL, WARN))

    @property
    def is_great(self) -> bool:
        """Great business = zero hard FAILs among applicable checks."""
        return self.n_fail == 0 and self.applicable >= 6

    @property
    def score(self) -> float:
        if not self.applicable:
            return 0.0
        return round(100 * (self.n_pass + 0.5 * self.n_warn) / self.applicable, 1)

    def to_dict(self):
        return {
            "ticker": self.ticker, "company": self.company,
            "sector": self.sector, "industry": self.industry,
            "country": self.country, "market_cap": self.market_cap,
            "company_type": self.company_type,
            "score": self.score, "is_great": self.is_great,
            "n_pass": self.n_pass, "n_fail": self.n_fail, "n_warn": self.n_warn,
            "checks": [{"name": c.name, "status": c.status,
                        "value": c.value, "detail": c.detail} for c in self.checks],
            "moat_hints": self.moat_hints,
            "error": self.error,
        }


# ---------------------------------------------------------------- checks

def _fmt_pct(x: Optional[float]) -> str:
    return f"{x*100:.1f}%" if x is not None else "n/a"


def run_checks(meta: Dict, data: Dict[str, Dict]) -> ScanResult:
    res = ScanResult(
        ticker=meta["ticker"], company=meta.get("company", ""),
        sector=meta.get("sector", ""), industry=meta.get("industry", ""),
        country=meta.get("country", ""), market_cap=meta.get("market_cap", ""),
    )
    ctype = classify(res.sector, res.industry)
    res.company_type = ctype

    inc = data.get("income") or {}
    bal = data.get("balance") or {}
    cf = data.get("cashflow") or {}
    rat = data.get("ratios") or {}

    def add(name, status, value=None, detail=""):
        res.checks.append(CheckResult(name, status, value, detail))

    # ---- 1. Sales consistently increasing
    rev = _series(inc, "revenue")
    ok = _consistently_increasing(rev)
    add("Sales increasing (5y)",
        NA if ok is None else (PASS if ok else FAIL),
        _fmt_pct(_cagr(rev)) + " CAGR" if _cagr(rev) is not None else None,
        "Revenue up over window with ≤1 down year")

    # ---- 2. Net income consistently increasing (operating income fallback)
    ni = _series(inc, "netIncome")
    ok_ni = _consistently_increasing(ni)
    if ok_ni is False:
        oi = _series(inc, "operatingIncome")
        ok_oi = _consistently_increasing(oi)
        if ok_oi:
            add("Net income increasing (5y)", WARN,
                _fmt_pct(_cagr(ni)) + " CAGR",
                "Net income choppy but OPERATING income consistently rising "
                "(course-approved fallback: excludes one-off items)")
        else:
            add("Net income increasing (5y)", FAIL,
                _fmt_pct(_cagr(ni)) + " CAGR" if _cagr(ni) is not None else None,
                "Neither net income nor operating income consistently rising")
    else:
        add("Net income increasing (5y)",
            NA if ok_ni is None else PASS,
            _fmt_pct(_cagr(ni)) + " CAGR" if _cagr(ni) is not None else None)

    # ---- 3. CFO consistently increasing (n/a: banks/property/commodity)
    ocf = _series(cf, "ncfo")
    if ctype in ("financial", "property", "commodity"):
        add("CFO increasing (5y)", NA, None,
            f"Not applicable to {ctype} companies per VMI (inconsistent by nature)")
    else:
        ok = _consistently_increasing(ocf)
        add("CFO increasing (5y)",
            NA if ok is None else (PASS if ok else FAIL),
            _fmt_pct(_cagr(ocf)) + " CAGR" if _cagr(ocf) is not None else None)

    # ---- 4. FCF consistently positive
    fcf = _series(cf, "fcf") or _series(inc, "fcf")
    if ctype == "financial":
        add("FCF positive", NA, None, "Capex/FCF not meaningful for financials")
    else:
        vals = [v for v in fcf if v is not None]
        if not vals:
            add("FCF positive", NA)
        else:
            neg = sum(1 for v in vals if v < 0)
            latest_neg = vals[0] < 0
            if neg == 0:
                add("FCF positive", PASS, "all 5y positive")
            elif latest_neg:
                add("FCF positive", FAIL, f"{neg}/{len(vals)} years negative (incl. latest)")
            else:
                add("FCF positive", WARN, f"{neg}/{len(vals)} years negative (latest positive)")

    # ---- 5/6. Gross & net margin consistent or increasing
    gm = _series(inc, "grossMargin")
    ok = _consistent_or_increasing_margin(gm)
    add("Gross margin stable/up (5y)",
        NA if ok is None else (PASS if ok else FAIL),
        f"latest {_oldest_first(gm)[-1]:.1f}%" if _oldest_first(gm) else None)

    nm = _series(inc, "profitMargin")
    ok = _consistent_or_increasing_margin(nm)
    add("Net margin stable/up (5y)",
        NA if ok is None else (PASS if ok else FAIL),
        f"latest {_oldest_first(nm)[-1]:.1f}%" if _oldest_first(nm) else None)

    # ---- 7. ROE >= 12% (5y average AND latest)
    roe = _series(rat, "roe")
    roe_avg = _avg(roe)
    roe_latest = roe[0] if roe else None
    equity_latest = _latest(bal, "equity")
    if roe_avg is None:
        add("ROE ≥ 12%", NA)
    elif equity_latest is not None and equity_latest < 0:
        add("ROE ≥ 12%", WARN, "equity negative",
            "Negative shareholder equity (often from buybacks, e.g. MCD/YUM) — "
            "ROE meaningless; judge manually per course caveat")
    elif roe_avg >= 12 and (roe_latest or 0) >= 12:
        add("ROE ≥ 12%", PASS, f"5y avg {roe_avg:.1f}%, latest {roe_latest:.1f}%")
    elif roe_avg >= 10:
        add("ROE ≥ 12%", WARN, f"5y avg {roe_avg:.1f}%, latest {roe_latest:.1f}%",
            "Between the 10% screen floor and the 12-15% target")
    else:
        add("ROE ≥ 12%", FAIL, f"5y avg {roe_avg:.1f}%")

    # ---- 8. ROIC >= 12% (n/a banks)
    if ctype == "financial":
        add("ROIC ≥ 12%", NA, None, "Not applicable to banks/financials per VMI")
    else:
        roic = _series(rat, "roic")
        roic_avg = _avg(roic)
        roic_latest = roic[0] if roic else None
        if roic_avg is None:
            add("ROIC ≥ 12%", NA)
        elif roic_avg >= 12 and (roic_latest or 0) >= 12:
            add("ROIC ≥ 12%", PASS, f"5y avg {roic_avg:.1f}%, latest {roic_latest:.1f}%")
        elif roic_avg >= 10:
            add("ROIC ≥ 12%", WARN, f"5y avg {roic_avg:.1f}%, latest {roic_latest:.1f}%")
        else:
            add("ROIC ≥ 12%", FAIL, f"5y avg {roic_avg:.1f}%")

    # ---- 9. Current ratio >= 1 (n/a financial/REIT)
    if ctype in ("financial", "reit"):
        add("Current ratio ≥ 1", NA, None, f"Not applicable to {ctype}")
    else:
        cr = _series(rat, "currentratio")
        cr_latest = cr[0] if cr else None
        if cr_latest is None:
            add("Current ratio ≥ 1", NA)
        elif cr_latest >= 1:
            add("Current ratio ≥ 1", PASS, f"{cr_latest:.2f}")
        elif cr_latest >= 0.8:
            unearned = _latest(bal, "balance_sheet_unearned_revenue")
            liabc = _latest(bal, "liabilitiesc")
            if unearned and liabc and unearned / liabc > 0.2:
                add("Current ratio ≥ 1", WARN, f"{cr_latest:.2f}",
                    "Below 1 but large deferred revenue in current liabilities "
                    "(course caveat: deliberate low current ratio)")
            else:
                add("Current ratio ≥ 1", WARN, f"{cr_latest:.2f}", "Slightly below 1")
        else:
            add("Current ratio ≥ 1", FAIL, f"{cr_latest:.2f}")

    # ---- 10. Debt/EBITDA <= 3 (n/a financial/REIT/property)
    if ctype in ("financial", "reit", "property"):
        add("Debt/EBITDA ≤ 3", NA, None, f"Not applicable to {ctype}")
    else:
        de = _series(rat, "debtebitda")
        de_latest = de[0] if de else None
        if de_latest is None:
            add("Debt/EBITDA ≤ 3", NA)
        elif de_latest <= 3:
            add("Debt/EBITDA ≤ 3", PASS, f"{de_latest:.2f}")
        elif de_latest <= 4:
            add("Debt/EBITDA ≤ 3", WARN, f"{de_latest:.2f}")
        else:
            add("Debt/EBITDA ≤ 3", FAIL, f"{de_latest:.2f}")

    # ---- 11. Debt servicing ratio < 30% (net interest expense / CFO)
    if ctype in ("financial", "reit", "property"):
        add("Debt servicing < 30%", NA, None, f"Not applicable to {ctype}")
    else:
        int_exp = _latest(inc, "income_statement_interest_expense")
        int_inc = _latest(inc, "interestIncome")
        ocf_latest = ocf[0] if ocf else None
        if ocf_latest is None or ocf_latest <= 0:
            add("Debt servicing < 30%", NA if ocf_latest is None else FAIL,
                None if ocf_latest is None else "CFO negative")
        elif int_exp is None:
            add("Debt servicing < 30%", PASS, "no interest expense reported",
                "No interest expense line — effectively debt-free or immaterial")
        else:
            net_int = abs(int_exp) - (int_inc or 0)
            if net_int <= 0:
                add("Debt servicing < 30%", PASS, "net interest income",
                    "Interest income exceeds interest expense")
            else:
                dsr = net_int / ocf_latest * 100
                if dsr < 30:
                    add("Debt servicing < 30%", PASS, f"{dsr:.1f}%")
                elif dsr < 40:
                    add("Debt servicing < 30%", WARN, f"{dsr:.1f}%")
                else:
                    add("Debt servicing < 30%", FAIL, f"{dsr:.1f}%")

    # ---- REIT gearing < 45%
    if ctype == "reit":
        debt = _latest(bal, "debt")
        assets = _latest(bal, "assets")
        if debt is not None and assets:
            g = debt / assets * 100
            if g < 40:
                add("REIT gearing < 45%", PASS, f"{g:.1f}%", "Under the safer 40% bar")
            elif g < 45:
                add("REIT gearing < 45%", WARN, f"{g:.1f}%", "Between 40% and 45%")
            else:
                add("REIT gearing < 45%", FAIL, f"{g:.1f}%")
        else:
            add("REIT gearing < 45%", NA)

    # ---- Bank-specific: CET1 / NPL not on free sources
    if ctype == "financial":
        add("Bank CET1 > 10% / NPL < 5%", NA, None,
            "Not available from free sources — check the bank's investor "
            "relations / Basel III Pillar 3 disclosures manually")

    # ---- 12. Receivables growing no faster than sales
    rec = _series(bal, "balance_sheet_total_trade_receivables") or \
        _series(bal, "balance_sheet_accounts_receivable")
    rev_cagr = _cagr(rev)
    rec_cagr = _cagr(rec)
    if ctype == "financial":
        add("Receivables ≤ sales growth", NA, None, "Not meaningful for financials")
    elif rev_cagr is None or rec_cagr is None:
        add("Receivables ≤ sales growth", NA, None,
            "Receivables not reported or insufficient history")
    elif rec_cagr <= rev_cagr + 0.03:  # 3pp tolerance
        add("Receivables ≤ sales growth", PASS,
            f"recv {_fmt_pct(rec_cagr)} vs sales {_fmt_pct(rev_cagr)}")
    elif rec_cagr <= rev_cagr + 0.10:
        add("Receivables ≤ sales growth", WARN,
            f"recv {_fmt_pct(rec_cagr)} vs sales {_fmt_pct(rev_cagr)}",
            "Receivables outgrowing sales modestly — monitor")
    else:
        add("Receivables ≤ sales growth", FAIL,
            f"recv {_fmt_pct(rec_cagr)} vs sales {_fmt_pct(rev_cagr)}",
            "Red flag per course (possible channel stuffing)")

    # ---- 13. Positive projected growth — guaranteed by Finviz pre-filter
    add("Positive projected growth", PASS, "EPS next-5Y estimate > 0",
        "Enforced by the Finviz pre-screen (analyst estimates)")

    # ---------------- Moat hints (informational only — user decides) ----
    gm_latest = _oldest_first(gm)[-1] if _oldest_first(gm) else None
    om = _series(inc, "operatingMargin")
    om_latest = _oldest_first(om)[-1] if _oldest_first(om) else None
    buyback = _series(rat, "buybackyield")
    bb_avg = _avg(buyback)
    roic_series = _series(rat, "roic")
    roic_all_high = None
    vals = [v for v in roic_series[:5] if v is not None]
    if vals:
        roic_all_high = all(v >= 15 for v in vals)
    hints = {}
    if gm_latest is not None:
        hints["gross_margin"] = f"{gm_latest:.1f}%" + (
            " (high — pricing power?)" if gm_latest >= 50 else "")
    if om_latest is not None:
        hints["operating_margin"] = f"{om_latest:.1f}%" + (
            " (high)" if om_latest >= 25 else "")
    if roic_all_high is not None:
        hints["roic_persistence"] = ("ROIC ≥ 15% every year for 5y — strong moat signal"
                                     if roic_all_high else "ROIC not uniformly ≥ 15%")
    if bb_avg is not None:
        hints["buybacks"] = (f"avg buyback yield {bb_avg:.1f}%/yr — self-financing, "
                             "shareholder friendly" if bb_avg > 0.5
                             else f"avg buyback/dilution {bb_avg:.1f}%/yr")
    hints["verdict"] = ("Moat assessment left to user discretion per VMI — "
                        "check brand, switching costs, network effect, "
                        "barriers to entry, economies of scale")
    res.moat_hints = hints
    return res
