"""VMI 'Great Business' checks (fundamentals only — no valuation, no TA).

Implements Adam Khoo's checklist from Lessons 4 & 7 / Quick Reference:

Profitability
  1. Sales revenue consistently increasing (up to 15y durable trend)
  2. Net income (or operating income fallback) consistently increasing (15y)
  3. Cash Flow from Operations consistently increasing (15y)
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
  - Docs describe Sales/Net-Income/CFO consistency as a long-term test
    ("5 years+" — a floor, not a fixed window). Per user instruction these
    trends use all available history, up to 15 years.
  - ROIC and receivables-vs-sales have no fixed window -> 10y default.

Company-type policy (narrowed per user instruction — only exclude types
whose great-business evaluation NEEDS data we cannot get):
  * ETFs — funds, not operating companies; the checklist doesn't apply.
  * Banks — need CET1 ratio (> 10%) and NPL data; no free source wired.
  * REITs — need gearing (< 45%) / FFO treatment; not wired.
  Everything else stays IN (incl. non-bank financials like MA/SPGI,
  property developers, commodity producers). The course-defined
  non-applicable debt checks are marked NA, not the whole company excluded.

Threshold calibration (validated against Adam's current top-stock list —
AAPL/MSFT/NVDA/META/... must all classify as great businesses):
  * ROE/ROIC 10-12% = WARN not FAIL (the course's own Finviz screen uses
    ROE > 10% as its floor; targets stated as "12%-15%+").
  * Debt/EBITDA 3-6x = WARN (course case-study material includes ~5.6x);
    > 6x = FAIL.
  * "Consistently increasing" = durable regression trend, not a max-dip
    count (see _consistently_increasing).
  * Receivables outgrowing sales = WARN (course calls it a red flag to
    investigate, not an automatic disqualifier).
  * Growth-stage loss->profit transitions (PANW/CRWD-style) soften
    historical-average FAILs to WARN when the smoothed trend is improving.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PASS, FAIL, WARN, NA = "PASS", "FAIL", "WARN", "NA"

# Docs give an explicit 5-year window for exactly these three checks.
WINDOW_5Y = 5
# Everything else that needs an averaging/consistency window but has no
# explicit number in the docs defaults to 10 years (user instruction).
WINDOW_10Y = 10
# User-selected long history for the Sales/NI/CFO consistency trends.
WINDOW_TREND = 15
# Multi-window rule (user instruction): every trend/average check is tried
# over 20y, 15y and 10y and PASSES if ANY window passes. A 5y-only pass is
# gated behind the accept_5y toggle — by default 5y alone is NOT enough
# and yields WARN instead.
LONG_WINDOWS = (20, 15, 10)

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


def _series(fd: Dict, key: str, max_n: int = 20, scale_percent: bool = False) -> List[Optional[float]]:
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
                             min_points: int = 5) -> Optional[bool]:
    """Durable upward trend test — NOT literal annual monotonicity.

    A dip-count rule gets stricter merely because more history is available:
    AAPL has four down-CFO years across 15 years while CFO still compounded
    ~8%/yr — clearly a great business. All conditions must hold:
      * positive least-squares slope across all available annual points;
      * latest value above the oldest value;
      * recent 3-year average above the earliest 3-year average;
      * at least half of year-to-year moves are increases.
    Accepts normal operating volatility without a hidden CAGR hurdle;
    rejects flat, deteriorating and one-year-spike series; handles
    loss-to-profit transitions where CAGR is mathematically undefined.
    """
    s = _oldest_first(s_newest_first)
    if len(s) < min_points:
        return None
    n = len(s)
    mean_x = (n - 1) / 2
    mean_y = sum(s) / n
    slope_num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(s))
    up_moves = sum(1 for a, b in zip(s, s[1:]) if b > a)
    edge = min(3, n // 2)
    early_avg = sum(s[:edge]) / edge
    recent_avg = sum(s[-edge:]) / edge
    return (slope_num > 0 and s[-1] > s[0] and recent_avg > early_avg
            and up_moves / (n - 1) >= 0.5)


def _multi_window_trend(s_newest: List[Optional[float]], accept_5y: bool):
    """Durable-trend test over 20/15/10y — PASS if ANY long window passes.
    Falls back to 5y: PASS only when accept_5y, else WARN (review flag).
    Returns (status, window_used, cagr_of_that_window)."""
    avail = sum(1 for v in s_newest if v is not None)
    tried_long = False
    for w in LONG_WINDOWS:
        # A long-window claim needs ≥8 annual points (or all we have when
        # the series is shorter than the window).
        if avail < min(w, 8):
            continue
        tried_long = True
        win = _window(s_newest, w)
        if _consistently_increasing(win, min_points=8):
            return PASS, w, _cagr(win)
    win5 = _window(s_newest, 5)
    ok5 = _consistently_increasing(win5, min_points=5)
    if ok5:
        return (PASS if accept_5y else WARN), 5, _cagr(win5)
    if not tried_long and ok5 is None:
        return NA, None, None
    return FAIL, None, _cagr(_window(s_newest, 15))


def _multi_window_avg(s_newest: List[Optional[float]], threshold: float,
                      accept_5y: bool):
    """Average-threshold test over 20/15/10y — PASS if ANY long-window
    average clears the threshold; 5y-only clearance gated by accept_5y.
    Returns (status, window_used, avg_of_that_window)."""
    best_long = None
    for w in LONG_WINDOWS:
        vals = [v for v in _window(s_newest, w) if v is not None]
        if len(vals) < min(w, 6) or len(vals) <= 5:
            continue  # degenerate: would be the same data as the 5y test
        avg = sum(vals) / len(vals)
        if best_long is None or avg > best_long[1]:
            best_long = (w, avg)
        if avg >= threshold:
            return PASS, w, avg
    vals5 = [v for v in _window(s_newest, 5) if v is not None]
    avg5 = sum(vals5) / len(vals5) if vals5 else None
    if avg5 is not None and avg5 >= threshold:
        return (PASS if accept_5y else WARN), 5, avg5
    if avg5 is None and best_long is None:
        return NA, None, None
    if best_long is not None:
        return FAIL, best_long[0], best_long[1]
    return FAIL, 5, avg5


def _improving_transition(s_newest_first: List[Optional[float]],
                          min_points: int = 5) -> Optional[bool]:
    """Whether a loss/choppy series is clearly improving on a smoothed basis
    (positive slope + recent avg above early avg). Softens growth-stage
    companies' historical-average FAILs to WARN."""
    s = _oldest_first(s_newest_first)
    if len(s) < min_points:
        return None
    n = len(s)
    mean_x = (n - 1) / 2
    mean_y = sum(s) / n
    slope_num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(s))
    edge = min(3, n // 2)
    return slope_num > 0 and sum(s[-edge:]) / edge > sum(s[:edge]) / edge


def _paired_cagrs(a_newest: List[Optional[float]],
                  b_newest: List[Optional[float]], n: int):
    """CAGRs over identical fiscal columns only, so two series with
    different missing-data patterns aren't compared over mismatched spans."""
    pairs = [(a, b) for a, b in zip(a_newest[:n], b_newest[:n])
             if a is not None and b is not None]
    if len(pairs) < 2:
        return None, None
    return _cagr([x[0] for x in pairs]), _cagr([x[1] for x in pairs])


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
    """Value for the LATEST fiscal column only — never fall back to an older
    non-null value (mixing current debt with stale EBITDA once gave TMO a
    fabricated Debt/EBITDA of 6.5 from mismatched years)."""
    s = _series(fd, key, scale_percent=scale_percent)
    return s[0] if s else None


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

FINANCIAL_INDUSTRY_HINTS = (
    "insurance", "capital markets", "asset management", "financial data",
    "credit services", "mortgage", "financial conglomerates",
    "consumer finance", "financial exchanges", "brokerage",
    # Managed-care insurers (UNH, ELV, CI, HUM): premium/claims float makes
    # current liabilities structurally large — the same insurance-float
    # balance-sheet reality the course's debt-check exception covers.
    "managed health care", "healthcare plans", "health care plans",
)
BANK_HINTS = ("bank", "banks", "banking", "thrifts")
REIT_HINT = "reit"
PROPERTY_HINTS = ("real estate development", "real estate services",
                  "real estate - development", "real estate management")
COMMODITY_HINTS = ("oil", "gas", "coal", "gold", "silver", "copper", "steel",
                   "aluminum", "mining", "chemicals", "agricultural inputs",
                   "metals & mining")


def classify(sector: str, industry: str, asset_type: str = "") -> str:
    """Return one of: etf, bank, reit, financial, property, commodity, standard.

    A broad "Financials" GICS sector label is NOT an exclusion — Mastercard
    and S&P Global are operating businesses fully evaluable with the
    standard checklist. Only types whose evaluation requires data we don't
    have (ETF holdings, bank CET1/NPL, REIT gearing/FFO) are excluded.
    """
    s = (sector or "").lower().strip()
    i = (industry or "").lower()
    if (asset_type or "").lower() in ("etf", "fund"):
        return "etf"
    if REIT_HINT in i:
        return "reit"
    if any(h in i for h in BANK_HINTS):
        return "bank"
    if any(h in i for h in PROPERTY_HINTS):
        return "property"
    if any(h in i for h in COMMODITY_HINTS):
        return "commodity"
    if s in ("financial", "financials") or any(h in i for h in FINANCIAL_INDUSTRY_HINTS):
        return "financial"
    return "standard"


# Only types whose great-business evaluation NEEDS unavailable data are
# excluded: ETFs (not companies), banks (CET1/NPL), REITs (gearing/FFO).
EXCLUDED_TYPES = {"etf", "bank", "reit"}


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
    data_source: str = ""
    checks: List[CheckResult] = field(default_factory=list)
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
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

    # Core checks that must be hard PASSes (not merely WARN) for the GREAT
    # verdict. Calibrated against the user's benchmark set (UNH/AAPL/MSFT/
    # GOOGL/NVDA/V/CAT all retained): this cuts S&P500 "greats" from 183 to
    # ~88 without dropping any benchmark. NA still doesn't disqualify —
    # a missing analyst estimate or non-applicable check isn't evidence of
    # a bad business.
    CORE_CHECKS = (
        "Sales increasing (multi-window)",
        "Net income increasing (multi-window)",
        "CFO increasing (multi-window)",
        "ROE ≥ 12%",
        "ROIC ≥ 12%",
        "Positive projected growth",
    )

    @property
    def is_great(self) -> bool:
        """Great business = zero hard FAILs anywhere, at most 2 review
        WARNs, every core check a hard PASS (WARN on a core check is not
        enough; NA is tolerated — absence of data never disqualifies), and
        enough applicable checks for a meaningful verdict."""
        if self.n_fail != 0 or self.n_warn > 2 or self.applicable < 8:
            return False
        status = {c.name: c.status for c in self.checks}
        return all(status.get(n) in (PASS, NA) for n in self.CORE_CHECKS)

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
            "company_type": self.company_type, "data_source": self.data_source,
            "score": self.score, "is_great": self.is_great,
            "n_pass": self.n_pass, "n_fail": self.n_fail, "n_warn": self.n_warn,
            "checks": [{"name": c.name, "status": c.status,
                        "value": c.value, "detail": c.detail} for c in self.checks],
            "metrics": self.metrics,
            "moat_hints": self.moat_hints,
            "error": self.error,
            "excluded": self.excluded,
            "exclusion_reason": self.exclusion_reason,
        }


# ---------------------------------------------------------------- checks

def _fmt_pct(x: Optional[float]) -> str:
    return f"{x*100:.1f}%" if x is not None else "n/a"


def run_checks(meta: Dict, data: Dict[str, Dict],
               growth_estimate: Optional[Dict] = None,
               require_5y_only_pass: bool = False,
               has_growth_prefilter: bool = False) -> ScanResult:
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
        data_source=meta.get("data_source", ""),
    )
    ctype = classify(res.sector, res.industry, meta.get("asset_type", ""))
    res.company_type = ctype

    inc = data.get("income") or {}
    bal = data.get("balance") or {}
    cf = data.get("cashflow") or {}
    rat = data.get("ratios") or {}

    def add(name, status, value=None, detail=""):
        res.checks.append(CheckResult(name, status, value, detail))

    accept_5y = not require_5y_only_pass

    def _wl(w):
        return f"{w}y" if w else ""

    # ---- 1. Sales consistently increasing — multi-window (20/15/10y
    # any-pass; a 5y-only pass is WARN unless the UI toggle allows it)
    rev = _first_present(_series(inc, "revenue"))
    st, w, cagr = _multi_window_trend(rev, accept_5y)
    res.metrics["rev_cagr_5y"] = _cagr(_window(rev, 5))
    res.metrics["rev_cagr_10y"] = _cagr(_window(rev, 10))
    res.metrics["rev_cagr_15y"] = _cagr(_window(rev, 15))
    add("Sales increasing (multi-window)", st,
        (_fmt_pct(cagr) + f" CAGR ({_wl(w)})") if cagr is not None else None,
        "Durable-trend test over ANY of 20/15/10y"
        + ("; 5y alone also accepted" if accept_5y else "; 5y alone → WARN"))

    # ---- 2. Net income consistently increasing — multi-window, with the
    # course-approved operating-income fallback
    ni = _first_present(_series(inc, "netIncome"), _series(inc, "cf_netIncome"))
    oi = _series(inc, "operatingIncome")
    ni_w = _window(ni, WINDOW_TREND)
    oi_w = _window(oi, WINDOW_TREND)
    res.metrics["ni_cagr_5y"] = _cagr(_window(ni, 5))
    res.metrics["ni_cagr_10y"] = _cagr(_window(ni, 10))
    res.metrics["ni_cagr_15y"] = _cagr(_window(ni, 15))
    st_ni, w_ni, cagr_ni = _multi_window_trend(ni, accept_5y)
    if st_ni == FAIL:
        st_oi, w_oi, _c = _multi_window_trend(oi, accept_5y)
        if st_oi == PASS:
            add("Net income increasing (multi-window)", WARN,
                (_fmt_pct(cagr_ni) + " CAGR") if cagr_ni is not None else None,
                "Net income choppy but OPERATING income consistently rising "
                f"({_wl(w_oi)}) — course-approved fallback (one-off items)")
        elif _improving_transition(ni_w) or _improving_transition(oi_w):
            add("Net income increasing (multi-window)", WARN,
                (_fmt_pct(cagr_ni) + " CAGR") if cagr_ni is not None else None,
                "Profitability improving on a smoothed long-term basis but not "
                "yet meeting the mature-company consistency test (growth-stage)")
        else:
            add("Net income increasing (multi-window)", FAIL,
                (_fmt_pct(cagr_ni) + " CAGR") if cagr_ni is not None else None,
                "Neither net income nor operating income consistently rising "
                "over any of 20/15/10y (or 5y)")
    else:
        add("Net income increasing (multi-window)", st_ni,
            (_fmt_pct(cagr_ni) + f" CAGR ({_wl(w_ni)})") if cagr_ni is not None else None,
            "5y-only pass — review flag" if st_ni == WARN else "")

    # ---- 3. CFO consistently increasing — multi-window
    ocf = _first_present(_series(cf, "ncfo"))
    st, w, cagr = _multi_window_trend(ocf, accept_5y)
    res.metrics["cfo_cagr_10y"] = _cagr(_window(ocf, 10))
    add("CFO increasing (multi-window)", st,
        (_fmt_pct(cagr) + f" CAGR ({_wl(w)})") if cagr is not None else None,
        "5y-only pass — review flag" if st == WARN else "")

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
    gm_vals = _oldest_first(gm_w)
    if ok is None:
        gm_status = NA
    elif ok:
        gm_status = PASS
    else:
        gm_avg = sum(gm_vals) / len(gm_vals) if gm_vals else 0
        gm_status = WARN if gm_vals and gm_vals[-1] > 0 and gm_vals[-1] >= gm_avg * 0.5 else FAIL
    add(f"Gross margin stable/up ({WINDOW_5Y}y)", gm_status,
        f"latest {gm_vals[-1]:.1f}%" if gm_vals else None,
        "Positive but compressed vs 5y average — review whether temporary"
        if gm_status == WARN else "")

    nm = _first_present(_series(inc, "profitMargin"), _series(rat, "profitMargin"))
    nm_w = _window(nm, WINDOW_5Y)
    ok = _consistent_or_increasing_margin(nm_w)
    nm_vals = _oldest_first(nm_w)
    growth_transition = bool(_improving_transition(ni_w) or _improving_transition(oi_w))
    if ok is None:
        nm_status = NA
    elif ok:
        nm_status = PASS
    else:
        nm_avg = sum(nm_vals) / len(nm_vals) if nm_vals else 0
        nm_status = WARN if growth_transition or (
            nm_vals and nm_vals[-1] > 0 and nm_vals[-1] >= nm_avg * 0.5
        ) else FAIL
    add(f"Net margin stable/up ({WINDOW_5Y}y)", nm_status,
        f"latest {nm_vals[-1]:.1f}%" if nm_vals else None,
        "Positive but compressed / growth-transition — review whether temporary"
        if nm_status == WARN else "")

    # ---- 7. ROE >= 12% — multi-window average (20/15/10y any-pass; 5y gated)
    roe = _series(rat, "roe")
    roe_latest = next((v for v in roe if v is not None), None)
    st, w, roe_avg = _multi_window_avg(roe, 12.0, accept_5y)
    if st == NA:
        add("ROE ≥ 12%", NA)
    elif any(v is not None and v < 0
             for v in _window(_series(bal, "equity"), WINDOW_10Y)):
        add("ROE ≥ 12%", WARN, "negative equity in window",
            "ROE distorted by negative shareholder equity (often buybacks, "
            "e.g. MCD/YUM/AZO) — course explicitly says judge manually")
    else:
        val = f"{_wl(w)} avg {roe_avg:.1f}%" if roe_avg is not None else None
        if val and roe_latest is not None:
            val += f", latest {roe_latest:.1f}%"
        if st == FAIL and roe_avg is not None and roe_avg >= 10:
            add("ROE ≥ 12%", WARN, val,
                "10-12%: between the course's Finviz floor (>10%) and the "
                "12-15% target")
        elif st == FAIL and growth_transition:
            add("ROE ≥ 12%", WARN, val,
                "Growth-stage profitability improving; historical ROE not yet "
                "representative")
        else:
            add("ROE ≥ 12%", st, val,
                "5y-only clearance — review flag" if st == WARN and w == 5 else "")

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
            v = ebit * (1 - tax_rate) / invested_capital * 100
            # Near-zero invested capital (heavy-buyback balance sheets)
            # produces absurd magnitudes — not a real return; skip.
            roic_computed.append(v if abs(v) <= 500 else None)
        else:
            roic_computed.append(None)
    roic_w = _window(roic_computed, WINDOW_10Y)
    roic_latest = next((v for v in roic_computed if v is not None), None)
    st, w, roic_avg = _multi_window_avg(roic_computed, 12.0, accept_5y)
    if st == NA:
        add("ROIC ≥ 12%", NA, None,
            "Insufficient data to compute EBIT x (1-tax) / (Equity+Debt-Cash)")
    else:
        val = f"{_wl(w)} avg {roic_avg:.1f}%" if roic_avg is not None else None
        if val and roic_latest is not None:
            val += f", latest {roic_latest:.1f}%"
        if st == FAIL and roic_avg is not None and roic_avg >= 10:
            add("ROIC ≥ 12%", WARN, val,
                "10-12%: below the 12-15% target but at/above the course's own "
                "10% screen floor — review flag, not a hard fail")
        elif st == FAIL and growth_transition:
            add("ROIC ≥ 12%", WARN, val,
                "Growth-stage profitability transition makes the historical "
                "average unrepresentative; review normalized ROIC manually")
        else:
            add("ROIC ≥ 12%", st, val,
                "5y-only clearance — review flag" if st == WARN and w == 5 else "")

    # ---- 9. Current ratio >= 1
    # Course: the standard debt checks aren't apples-to-apples for financial
    # firms / property developers / commodity producers (structurally
    # different balance sheets). Keep the company; mark these checks NA.
    debt_checks_apply = ctype not in {"financial", "property", "commodity"}
    cr = _series(rat, "currentRatio") or _series(rat, "currentratio")
    cr_latest = cr[0] if cr else None
    if not debt_checks_apply:
        add("Current ratio ≥ 1", NA, None,
            f"Standard debt test not applicable to {ctype} companies under VMI rules")
    elif cr_latest is None:
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
    if not debt_checks_apply:
        add("Debt/EBITDA ≤ 3", NA, None,
            f"Standard debt test not applicable to {ctype} companies under VMI rules")
    elif de_latest is None:
        add("Debt/EBITDA ≤ 3", NA)
    elif de_latest <= 3:
        add("Debt/EBITDA ≤ 3", PASS, f"{de_latest:.2f}")
    elif de_latest <= 6:
        add("Debt/EBITDA ≤ 3", WARN, f"{de_latest:.2f}",
            "Above the preferred ≤3 target; course case-study material includes "
            "a ~5.6x example, so this is a review flag rather than auto-reject")
    else:
        add("Debt/EBITDA ≤ 3", FAIL, f"{de_latest:.2f}")

    # ---- 11. Debt servicing ratio < 30% (net interest expense / CFO)
    int_exp = _latest(inc, "income_statement_interest_expense")
    int_inc = _latest(inc, "interestIncome")
    ocf_latest = ocf[0] if ocf else None
    if not debt_checks_apply:
        add("Debt servicing < 30%", NA, None,
            f"Standard debt test not applicable to {ctype} companies under VMI rules")
    elif ocf_latest is None or ocf_latest <= 0:
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
    rev_cagr, rec_cagr = _paired_cagrs(rev, rec, WINDOW_10Y)
    if rev_cagr is None or rec_cagr is None:
        add("Receivables ≤ sales growth", NA, None,
            "Receivables not reported or insufficient history")
    elif rec_cagr <= rev_cagr + 0.03:  # 3pp tolerance
        add("Receivables ≤ sales growth", PASS,
            f"recv {_fmt_pct(rec_cagr)} vs sales {_fmt_pct(rev_cagr)}")
    else:
        add("Receivables ≤ sales growth", WARN,
            f"recv {_fmt_pct(rec_cagr)} vs sales {_fmt_pct(rev_cagr)}",
            "Course red flag (possible channel stuffing), but not proof of a "
            "bad business — inspect customer/distribution changes manually")

    # ---- 13. Positive projected growth — analyst EPS estimates (finviz
    # bulk pull wired through scan.py). NA only when finviz genuinely has
    # no estimate; NA never disqualifies.
    g = growth_estimate or {}
    proj5 = g.get("eps_next_5y")
    proj1 = g.get("eps_next_y")
    res.metrics["proj_eps_next_5y"] = proj5
    res.metrics["proj_eps_next_y"] = proj1
    res.metrics["eps_past_5y"] = g.get("eps_past_5y")
    if proj5 is not None:
        if proj5 > 0:
            add("Positive projected growth", PASS,
                f"EPS next-5Y est {proj5:+.1f}%/yr"
                + (f", next-Y {proj1:+.1f}%" if proj1 is not None else ""))
        elif proj1 is not None and proj1 > 0:
            add("Positive projected growth", WARN,
                f"next-5Y {proj5:+.1f}%/yr but next-Y {proj1:+.1f}%",
                "Long-term estimate negative while next year positive — review")
        else:
            add("Positive projected growth", FAIL, f"EPS next-5Y est {proj5:+.1f}%/yr")
    elif proj1 is not None:
        add("Positive projected growth", PASS if proj1 > 0 else FAIL,
            f"EPS next-Y est {proj1:+.1f}% (no 5Y estimate)")
    elif has_growth_prefilter:
        add("Positive projected growth", PASS, "EPS next-5Y estimate > 0",
            "Enforced by the Finviz pre-screen (analyst estimates)")
    else:
        add("Positive projected growth", NA, None,
            "No analyst estimate available for this ticker — does not "
            "count against is_great")

    # ---------------- Intrinsic value via DCF (informational, no check) ----
    # Adam Khoo's "VMI IV Calculator (20 years)" — Discounted Free Cash Flow
    # method, replicated exactly from the course workbook formulas:
    #   FCF (latest annual, = CFO − Capex) projected 20 years, NO terminal:
    #     Yr 1-5  : analyst EPS-next-5Y growth estimate (clamped 0-20%;
    #               fallback: historical 10y FCF CAGR clamped the same)
    #     Yr 6-10 : same rate but capped at 15%   (MSFT example: 17.48%→15%)
    #     Yr 11-20: 4% flat                        (workbook F24)
    #   Discount rate = Rf + beta × MRP (CAPM); Rf/MRP are the 5y averages
    #     shipped in the workbook's "Discount Rate Data" sheet
    #     (market-risk-premia.com, updated 2026-03): Rf 3.608%, MRP 2.728%.
    #     Beta from finviz, clamped to the workbook's table range 0.8-1.6;
    #     1.0 assumed when unavailable.
    #   IV/share = PV/shares − total debt/share + (cash + ST invest)/share.
    price = g.get("price")
    shares = g.get("shares_outstanding")
    beta = g.get("beta")
    res.metrics["price"] = price
    fcf_latest = next((v for v in fcf if v is not None), None) if fcf else None
    iv_ps = None
    if fcf_latest is not None and fcf_latest > 0 and shares:
        RF, MRP = 0.03608, 0.02728
        b = min(max(beta if beta is not None else 1.0, 0.8), 1.6)
        disc = RF + b * MRP
        if proj5 is not None:
            g1 = min(max(proj5 / 100.0, 0.0), 0.20)
        else:
            hist = _cagr(_window(fcf, 10))
            g1 = min(max(hist if hist is not None else 0.0, 0.0), 0.20)
        g2 = min(g1, 0.15)
        G3 = 0.04
        pv, f = 0.0, fcf_latest
        for yr in range(1, 21):
            f *= (1 + (g1 if yr <= 5 else g2 if yr <= 10 else G3))
            pv += f / (1 + disc) ** yr
        iv_ps = pv / shares

        def _latest_bal(key):
            s = _series(bal, key)
            return next((v for v in s if v is not None), None) if s else None

        debt_total = (_latest_bal("shortTermDebt") or 0) + \
                     (_latest_bal("longTermDebt") or 0)
        cash_total = (_latest_bal("cash") or 0) + \
                     (_latest_bal("shortTermInvestments") or 0)
        iv_ps = iv_ps - debt_total / shares + cash_total / shares
        res.metrics["intrinsic_value"] = round(iv_ps, 2)
        res.metrics["dcf_growth_used"] = round(g1 * 100, 1)
        res.metrics["dcf_discount_rate"] = round(disc * 100, 2)
        if price and iv_ps > 0:
            # Positive = trading below IV (discount); negative = premium.
            res.metrics["discount_pct"] = round((iv_ps - price) / iv_ps * 100, 1)
        else:
            res.metrics["discount_pct"] = None
    else:
        res.metrics["intrinsic_value"] = None
        res.metrics["discount_pct"] = None

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
