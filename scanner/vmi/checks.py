"""VMI 'Great Business' checks (fundamentals only — no valuation, no TA).

Implements Adam Khoo's checklist from Lessons 4 & 7 / Quick Reference:

Profitability
  1. Sales revenue consistently increasing (10y default; docs say "5-10y")
  2. Net income (or operating income fallback) consistently increasing (10y)
  3. Cash Flow from Operations consistently increasing (10y)
  4. Free Cash Flow consistently positive
  5. Gross margin consistent or increasing (5y — EXPLICIT in docs)
  6. Net margin consistent or increasing (5y — EXPLICIT in docs)
  7. ROE >= 12% (5y — EXPLICIT in docs: "for the last 5 years")
  8. ROIC >= 12% (10y default — docs give NO window, only a target %)

Financial strength
  9.  Current ratio >= 1
  10. Debt / EBITDA <= 3
  11. Debt servicing ratio (net interest expense / CFO) < 30%

Management effectiveness
  12. Receivables growing no faster than sales (10y CAGR — docs give no
      window, only "must grow no faster than sales")

Forward-looking
  13. Positive projected growth (enforced by the Finviz pre-filter when
      that path is used; NA when scanning tickers directly without it)

Moat is NOT auto-decided (per user instruction): we compute *hints*
(margin level/stability, ROIC persistence, buybacks) and leave the final
moat call to user discretion.

Averaging-window policy (per explicit user instruction, verified against
VMI_Master_Reference.md):
  - Docs state an EXPLICIT 5-year window for: ROE ("for the last 5 years"),
    Gross margin ("over 5 years"), Net margin ("over 5 years").
    -> WINDOW_5Y = 5 is used for these three checks only.
  - Docs describe Sales/Net-Income/CFO "consistency" ambiguously as
    "5-10 years" (never pinned to a single number) and give NO window at
    all for ROIC, receivables-vs-sales growth, or CCC.
    -> Per the user's instruction ("if not defined use 10y"), these
       default to WINDOW_10Y = 10.

Company-type EXCLUSION (changed from earlier "exemption" design):
  Per explicit user instruction, REITs, banks/financial firms, property
  developers and commodity producers are now EXCLUDED from the scan
  entirely (not scored with per-check exemptions). `classify()` still
  runs so `scan.py` can filter them out before/after fetching data, and
  the type label is preserved on any result that slips through for
  transparency. See scan.py EXCLUDED_TYPES.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PASS, FAIL, WARN, NA = "PASS", "FAIL", "WARN", "NA"

# Docs give an explicit 5-year window for exactly these three checks.
WINDOW_5Y = 5
# Everything else that needs an averaging/consistency window but has no
# explicit number in the docs defaults to 10 years (user instruction).
WINDOW_10Y = 10

# stockanalysis.com serves these as fractions (0.25 = 25%); macrotrends.net
# serves the equivalent ratios already as percent. PERCENT_KEYS lists the
# *stockanalysis* field names that need the x100 fix; macrotrends fields are
# handled separately (see macrotrends.ALREADY_PERCENT) and never get scaled.
PERCENT_KEYS = {
    "roe", "roic", "roa", "roce", "grossMargin", "profitMargin",
    "operatingMargin", "buybackyield", "payoutratio", "dividendyield",
    "fcfMargin", "ebitdaMargin", "ebitMargin", "earningsyield", "fcfyield",
}

# ---------------------------------------------------------------- helpers


def _series(fd: Dict, key: str, max_n: int = 15, scale_percent: bool = False) -> List[Optional[float]]:
    """Newest-first numeric series for annual fiscal years (skip TTM col).

    Data shape: macrotrends.net dicts (clean annual columns only, ratio
    fields already expressed as percent, e.g. 46.9 = 46.9%). This is the
    only source wired into scan.py's pipeline; the shape also happens to
    be compatible with stockanalysis.com dicts (which include a leading
    TTM/current column, detected + stripped below via fiscalQuarter modal-
    label comparison) if that source is ever reintroduced as a fallback —
    in that case pass `scale_percent=True` explicitly for PERCENT_KEYS
    fields to convert stockanalysis's fractional convention (0.25 = 25%).

    Default `scale_percent=False` because macrotrends is the sole active
    source and its ratios need no scaling.
    """
    vals = fd.get(key)
    if not vals:
        return []
    years = fd.get("fiscalYear") or []
    out = list(zip(years, vals)) if years else [(None, v) for v in vals]

    fq = fd.get("fiscalQuarter") or []
    # The first column is TTM/Current when its quarter label differs from
    # the modal (most common) quarter label of the rest — annual rows all
    # carry the same fiscal-year-end quarter label. (macrotrends dicts have
    # no fiscalQuarter key, so this is a no-op for that source.)
    if len(fq) >= 3 and fq[0] is not None:
        rest = [str(q) for q in fq[1:] if q is not None]
        if rest:
            modal = max(set(rest), key=rest.count)
            if str(fq[0]) != modal:
                out = out[1:]

    vals2 = [v for (_y, v) in out][:max_n]
    scale = 100.0 if scale_percent else 1.0
    return [float(v) * scale if isinstance(v, (int, float)) else None for v in vals2]


def _oldest_first(s: List[Optional[float]]) -> List[float]:
    return [v for v in reversed(s) if v is not None]


def _window(s_newest_first: List[Optional[float]], n: int) -> List[Optional[float]]:
    """Truncate a newest-first series to the most recent `n` points."""
    return s_newest_first[:n]


def _consistently_increasing(s_newest_first: List[Optional[float]],
                             tolerance_dips: int = None,
                             min_points: int = 4) -> Optional[bool]:
    """True if series trends up with at most `tolerance_dips` down-years
    AND the last value is above the first (net growth over the window).

    `tolerance_dips` default (None) scales with the window length — one
    tolerated down-year per ~5 years of history (so a 10y window allows 2
    dips, a 5y window allows 1) — rather than a fixed constant, since a
    longer lookback should reasonably tolerate proportionally more
    cyclical dips (e.g. one bad year in a decade is not "inconsistent").
    """
    s = _oldest_first(s_newest_first)
    if len(s) < min_points:
        return None
    if tolerance_dips is None:
        tolerance_dips = max(1, len(s) // 5)
    dips = sum(1 for a, b in zip(s, s[1:]) if b < a)
    return dips <= tolerance_dips and s[-1] > s[0]


def _consistent_or_increasing_margin(s_newest_first: List[Optional[float]],
                                     max_drop_pp_frac: float = 0.15,
                                     min_points: int = 4) -> Optional[bool]:
    """Margins must be consistent or increasing: allow small wobble.
    Fails if the latest margin is >15% (relative) below the window average
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


def _latest(fd: Dict, key: str, scale_percent: bool = False) -> Optional[float]:
    s = _series(fd, key, scale_percent=scale_percent)
    for v in s:
        if v is not None:
            return v
    return None


def _avg(s: List[Optional[float]], n: int) -> Optional[float]:
    vals = [v for v in s[:n] if v is not None]
    return sum(vals) / len(vals) if vals else None


def _first_present(*series: List[Optional[float]]) -> List[Optional[float]]:
    """Return the first series that has at least one non-None value."""
    for s in series:
        if any(v is not None for v in s):
            return s
    return []


# ---------------------------------------------------------------- classify

# GICS sector strings (from the Wikipedia S&P500 table) that map straight
# to an excluded type — cheaper/more reliable than industry substring
# matching, used as the first check in classify().
GICS_SECTOR_EXCLUDE = {
    "financials": "financial",
    "real estate": "reit",
}

FINANCIAL_INDUSTRY_HINTS = (
    "bank", "insurance", "capital markets", "asset management",
    "financial data", "credit services", "mortgage", "financial conglomerates",
    "consumer finance", "financial exchanges",
)
REIT_HINT = "reit"
PROPERTY_HINTS = ("real estate development", "real estate services",
                  "real estate - development", "real estate management")
COMMODITY_HINTS = ("oil", "gas", "coal", "gold", "silver", "copper", "steel",
                   "aluminum", "mining", "chemicals", "agricultural inputs",
                   "metals & mining")


def classify(sector: str, industry: str) -> str:
    """Return one of: reit, financial, property, commodity, standard.

    Checks GICS sector first (cheap, reliable when sourced from the
    Wikipedia S&P500 table), then falls back to industry substring
    matching (needed for Finviz-sourced rows, which use free-text
    industry strings rather than GICS sector names).
    """
    s = (sector or "").lower().strip()
    i = (industry or "").lower()
    if s in GICS_SECTOR_EXCLUDE:
        return GICS_SECTOR_EXCLUDE[s]
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


# Types excluded from the scan entirely per explicit user instruction
# ("exclude REITS and banks and financial instruments and such, ie the
# businesses that have exceptions, for now"). scan.py filters these out
# before running deep checks; kept here too as a defensive double-check.
EXCLUDED_TYPES = {"reit", "financial", "property", "commodity"}


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
    excluded: bool = False
    exclusion_reason: str = ""

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
            "excluded": self.excluded,
            "exclusion_reason": self.exclusion_reason,
        }


# ---------------------------------------------------------------- checks

def _fmt_pct(x: Optional[float]) -> str:
    return f"{x*100:.1f}%" if x is not None else "n/a"


def run_checks(meta: Dict, data: Dict[str, Dict], has_growth_prefilter: bool = False) -> ScanResult:
    """Run the VMI fundamentals checklist.

    `data` must have keys "income", "balance", "cashflow", "ratios", each
    a dict of {field: [values newest-first]} — either macrotrends.net
    field names (already-percent ratios) or stockanalysis.com field names
    (fractional ratios, PERCENT_KEYS-scaled). Both shapes are supported
    transparently by `_series()`.

    `has_growth_prefilter`: True only when the ticker reached this point
    via a universe source that already enforced positive forward EPS
    growth (e.g. the optional Finviz pre-filter). When scanning the raw
    S&P500 universe directly (the default now), this is False and check
    #13 is marked NA rather than a rubber-stamp PASS.
    """
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

    # ---- 1. Sales consistently increasing (10y default — docs say "5-10y")
    rev = _first_present(_series(inc, "revenue"))
    rev_w = _window(rev, WINDOW_10Y)
    ok = _consistently_increasing(rev_w)
    add(f"Sales increasing ({WINDOW_10Y}y)",
        NA if ok is None else (PASS if ok else FAIL),
        _fmt_pct(_cagr(rev_w)) + " CAGR" if _cagr(rev_w) is not None else None,
        f"Revenue up over {WINDOW_10Y}y window with ≤1 down year "
        "(docs: '5-10 years', no fixed single number → defaulted to 10y)")

    # ---- 2. Net income consistently increasing (operating income fallback)
    ni = _first_present(_series(inc, "netIncome"), _series(inc, "cf_netIncome"))
    ni_w = _window(ni, WINDOW_10Y)
    ok_ni = _consistently_increasing(ni_w)
    if ok_ni is False:
        oi = _window(_series(inc, "operatingIncome"), WINDOW_10Y)
        ok_oi = _consistently_increasing(oi)
        if ok_oi:
            add(f"Net income increasing ({WINDOW_10Y}y)", WARN,
                _fmt_pct(_cagr(ni_w)) + " CAGR",
                "Net income choppy but OPERATING income consistently rising "
                "(course-approved fallback: excludes one-off items)")
        else:
            add(f"Net income increasing ({WINDOW_10Y}y)", FAIL,
                _fmt_pct(_cagr(ni_w)) + " CAGR" if _cagr(ni_w) is not None else None,
                "Neither net income nor operating income consistently rising")
    else:
        add(f"Net income increasing ({WINDOW_10Y}y)",
            NA if ok_ni is None else PASS,
            _fmt_pct(_cagr(ni_w)) + " CAGR" if _cagr(ni_w) is not None else None)

    # ---- 3. CFO consistently increasing (10y default)
    ocf = _first_present(_series(cf, "ncfo"))
    ocf_w = _window(ocf, WINDOW_10Y)
    ok = _consistently_increasing(ocf_w)
    add(f"CFO increasing ({WINDOW_10Y}y)",
        NA if ok is None else (PASS if ok else FAIL),
        _fmt_pct(_cagr(ocf_w)) + " CAGR" if _cagr(ocf_w) is not None else None)

    # ---- 4. FCF consistently positive (FCF = CFO - Capex)
    fcf_direct = _series(cf, "fcf") or _series(inc, "fcf")
    if any(v is not None for v in fcf_direct):
        fcf = fcf_direct
    else:
        capex = _series(cf, "capex")
        if ocf and capex and len(ocf) == len(capex):
            fcf = [(o - abs(c)) if (o is not None and c is not None) else None
                   for o, c in zip(ocf, capex)]
        else:
            fcf = []
    fcf_w = _window(fcf, WINDOW_10Y)
    vals = [v for v in fcf_w if v is not None]
    if not vals:
        add("FCF positive", NA)
    else:
        neg = sum(1 for v in vals if v < 0)
        latest_neg = vals[0] < 0
        if neg == 0:
            add("FCF positive", PASS, f"all {len(vals)}y positive")
        elif latest_neg:
            add("FCF positive", FAIL, f"{neg}/{len(vals)} years negative (incl. latest)")
        else:
            add("FCF positive", WARN, f"{neg}/{len(vals)} years negative (latest positive)")

    # ---- 5/6. Gross & net margin consistent or increasing (5y — EXPLICIT)
    gm = _first_present(_series(inc, "grossMargin"), _series(rat, "grossMargin"))
    gm_w = _window(gm, WINDOW_5Y)
    ok = _consistent_or_increasing_margin(gm_w)
    add(f"Gross margin stable/up ({WINDOW_5Y}y)",
        NA if ok is None else (PASS if ok else FAIL),
        f"latest {_oldest_first(gm_w)[-1]:.1f}%" if _oldest_first(gm_w) else None)

    nm = _first_present(_series(inc, "profitMargin"), _series(rat, "profitMargin"))
    nm_w = _window(nm, WINDOW_5Y)
    ok = _consistent_or_increasing_margin(nm_w)
    add(f"Net margin stable/up ({WINDOW_5Y}y)",
        NA if ok is None else (PASS if ok else FAIL),
        f"latest {_oldest_first(nm_w)[-1]:.1f}%" if _oldest_first(nm_w) else None)

    # ---- 7. ROE >= 12% (5y average AND latest — EXPLICIT "for the last 5 years")
    roe = _series(rat, "roe")
    roe_w = _window(roe, WINDOW_5Y)
    roe_avg = _avg(roe_w, WINDOW_5Y)
    roe_latest = roe_w[0] if roe_w else None
    equity_latest = _latest(bal, "equity")
    if roe_avg is None:
        add("ROE ≥ 12%", NA)
    elif equity_latest is not None and equity_latest < 0:
        add("ROE ≥ 12%", WARN, "equity negative",
            "Negative shareholder equity (often from buybacks, e.g. MCD/YUM) — "
            "ROE meaningless; judge manually per course caveat")
    elif roe_avg >= 12 and (roe_latest or 0) >= 12:
        add("ROE ≥ 12%", PASS, f"{WINDOW_5Y}y avg {roe_avg:.1f}%, latest {roe_latest:.1f}%")
    elif roe_avg >= 10:
        add("ROE ≥ 12%", WARN, f"{WINDOW_5Y}y avg {roe_avg:.1f}%, latest {roe_latest:.1f}%",
            "Between the 10% screen floor and the 12-15% target")
    else:
        add("ROE ≥ 12%", FAIL, f"{WINDOW_5Y}y avg {roe_avg:.1f}%")

    # ---- 8. ROIC >= 12% (10y default — docs give NO window, only a target %)
    # VMI formula (line 90): EBIT x (1 - tax rate) / (Equity + Debt - Cash).
    # Computed directly from raw statement components rather than relying
    # on macrotrends' generic "ROI" ratio (which is a different, broader
    # metric than VMI's specific ROIC definition).
    ebit_s = _series(inc, "ebit")
    pretax_s = _series(inc, "pretaxIncome")
    tax_s = _series(inc, "incomeTax")
    equity_s = _series(bal, "equity")
    debt_s = _series(bal, "longTermDebt")
    cash_s = _series(bal, "cash")
    n = min(len(ebit_s), len(pretax_s), len(tax_s), len(equity_s), len(debt_s), len(cash_s)) \
        if all([ebit_s, pretax_s, tax_s, equity_s, debt_s, cash_s]) else 0
    roic_computed: List[Optional[float]] = []
    for i in range(n):
        ebit, pretax, tax, eq, debt, cash = (ebit_s[i], pretax_s[i], tax_s[i],
                                              equity_s[i], debt_s[i], cash_s[i])
        if None in (ebit, pretax, tax, eq, debt, cash) or pretax == 0:
            roic_computed.append(None)
            continue
        tax_rate = tax / pretax
        invested_capital = eq + debt - cash
        if invested_capital and invested_capital > 0:
            roic_computed.append(ebit * (1 - tax_rate) / invested_capital * 100)
        else:
            roic_computed.append(None)
    roic_w = _window(roic_computed, WINDOW_10Y)
    roic_avg = _avg(roic_w, WINDOW_10Y)
    roic_latest = roic_w[0] if roic_w else None
    if roic_avg is None:
        add("ROIC ≥ 12%", NA, None,
            "Insufficient data to compute EBIT x (1-tax) / (Equity+Debt-Cash)")
    elif roic_avg >= 12 and (roic_latest or 0) >= 12:
        add("ROIC ≥ 12%", PASS, f"{WINDOW_10Y}y avg {roic_avg:.1f}%, latest {roic_latest:.1f}%")
    elif roic_avg >= 10:
        add("ROIC ≥ 12%", WARN, f"{WINDOW_10Y}y avg {roic_avg:.1f}%, latest {roic_latest:.1f}%")
    else:
        add("ROIC ≥ 12%", FAIL, f"{WINDOW_10Y}y avg {roic_avg:.1f}%")

    # ---- 9. Current ratio >= 1
    cr = _series(rat, "currentRatio") or _series(rat, "currentratio")
    cr_latest = cr[0] if cr else None
    if cr_latest is None:
        add("Current ratio ≥ 1", NA)
    elif cr_latest >= 1:
        add("Current ratio ≥ 1", PASS, f"{cr_latest:.2f}")
    elif cr_latest >= 0.8:
        add("Current ratio ≥ 1", WARN, f"{cr_latest:.2f}",
            "Slightly below 1 — check for deferred revenue in current "
            "liabilities (course caveat: deliberate low current ratio)")
    else:
        add("Current ratio ≥ 1", FAIL, f"{cr_latest:.2f}")

    # ---- 10. Debt/EBITDA <= 3
    debt = _latest(bal, "longTermDebt") if "longTermDebt" in bal else _latest(bal, "debt")
    ebitda = _latest(inc, "ebitda")
    de_direct = _series(rat, "debtebitda")
    if de_direct and de_direct[0] is not None:
        de_latest = de_direct[0]
    elif debt is not None and ebitda:
        de_latest = debt / ebitda
    else:
        de_latest = None
    if de_latest is None:
        add("Debt/EBITDA ≤ 3", NA)
    elif de_latest <= 3:
        add("Debt/EBITDA ≤ 3", PASS, f"{de_latest:.2f}")
    elif de_latest <= 4:
        add("Debt/EBITDA ≤ 3", WARN, f"{de_latest:.2f}")
    else:
        add("Debt/EBITDA ≤ 3", FAIL, f"{de_latest:.2f}")

    # ---- 11. Debt servicing ratio < 30% (net interest expense / CFO)
    int_exp = _latest(inc, "income_statement_interest_expense")
    int_inc = _latest(inc, "interestIncome")
    ocf_latest = ocf[0] if ocf else None
    if ocf_latest is None or ocf_latest <= 0:
        add("Debt servicing < 30%", NA if ocf_latest is None else FAIL,
            None if ocf_latest is None else "CFO negative")
    elif int_exp is None:
        add("Debt servicing < 30%", NA, None,
            "Interest expense not available from this free data source — "
            "check manually if debt levels look material")
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

    # ---- 12. Receivables growing no faster than sales (10y default — no window in docs)
    rec = _first_present(_series(bal, "receivables"),
                         _series(bal, "balance_sheet_total_trade_receivables"),
                         _series(bal, "balance_sheet_accounts_receivable"))
    rev_cagr = _cagr(_window(rev, WINDOW_10Y))
    rec_cagr = _cagr(_window(rec, WINDOW_10Y))
    if rev_cagr is None or rec_cagr is None:
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

    # ---- 13. Positive projected growth — only meaningful if a universe
    # pre-filter already enforced forward-analyst-estimate growth; when
    # scanning the raw S&P500 list directly we have no free forward-EPS
    # estimate source, so mark NA rather than rubber-stamping PASS.
    if has_growth_prefilter:
        add("Positive projected growth", PASS, "EPS next-5Y estimate > 0",
            "Enforced by the Finviz pre-screen (analyst estimates)")
    else:
        add("Positive projected growth", NA, None,
            "No free forward-estimate source wired in for direct-universe "
            "scans — check analyst estimates manually (this check does not "
            "count against is_great)")

    # ---------------- Moat hints (informational only — user decides) ----
    gm_latest = _oldest_first(gm)[-1] if _oldest_first(gm) else None
    om = _first_present(_series(rat, "operatingMargin"), _series(inc, "operatingMargin"))
    om_latest = _oldest_first(om)[-1] if _oldest_first(om) else None
    buyback = _series(rat, "buybackyield")
    bb_avg = _avg(buyback, WINDOW_5Y)
    roic_all_high = None
    rvals = [v for v in roic_w if v is not None]
    if rvals:
        roic_all_high = all(v >= 15 for v in rvals)
    hints = {}
    if gm_latest is not None:
        hints["gross_margin"] = f"{gm_latest:.1f}%" + (
            " (high — pricing power?)" if gm_latest >= 50 else "")
    if om_latest is not None:
        hints["operating_margin"] = f"{om_latest:.1f}%" + (
            " (high)" if om_latest >= 25 else "")
    if roic_all_high is not None:
        hints["roic_persistence"] = (f"ROIC ≥ 15% every year for {len(rvals)}y — strong moat signal"
                                     if roic_all_high else f"ROIC not uniformly ≥ 15% over {len(rvals)}y")
    if bb_avg is not None:
        hints["buybacks"] = (f"avg buyback yield {bb_avg:.1f}%/yr — self-financing, "
                             "shareholder friendly" if bb_avg > 0.5
                             else f"avg buyback/dilution {bb_avg:.1f}%/yr")
    hints["verdict"] = ("Moat assessment left to user discretion per VMI — "
                        "check brand, switching costs, network effect, "
                        "barriers to entry, economies of scale")
    res.moat_hints = hints
    return res
