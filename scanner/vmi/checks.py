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
# Long-window rule (user instruction, B-1): by DEFAULT every trend/average
# check must pass over the FULL 20-year window. A boolean toggle
# (any_long_window) re-enables the older lenient behavior where the check
# is tried over 20y, 15y and 10y and PASSES if ANY window passes.
# A 5y-only pass is separately gated behind the accept_5y toggle — by
# default 5y alone is NOT enough and yields WARN instead.
LONG_WINDOWS_ANY = (20, 15, 10)
LONG_WINDOWS_STRICT = (20,)


def _long_windows(any_long_window: bool):
    return LONG_WINDOWS_ANY if any_long_window else LONG_WINDOWS_STRICT

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


def _multi_window_trend(s_newest: List[Optional[float]], accept_5y: bool,
                        any_long_window: bool = False):
    """Durable-trend test over the long window(s).

    Default (any_long_window=False): only the 20-year window is tried —
    the check must pass on the full 20y history (all available annual
    points up to 20, minimum 8). With any_long_window=True the older
    lenient rule applies: 20/15/10y, PASS if ANY window passes.
    Falls back to 5y: PASS only when accept_5y, else WARN (review flag).
    Returns (status, window_used, cagr_of_that_window)."""
    avail = sum(1 for v in s_newest if v is not None)
    tried_long = False
    for w in _long_windows(any_long_window):
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
    return FAIL, None, _cagr(_window(s_newest,
                                     15 if any_long_window else 20))


def _multi_window_avg(s_newest: List[Optional[float]], threshold: float,
                      accept_5y: bool, any_long_window: bool = False):
    """Average-threshold test over the long window(s).

    Default (any_long_window=False): only the full 20-year average must
    clear the threshold. With any_long_window=True the older lenient
    rule applies: 20/15/10y, PASS if ANY window's average clears it.
    5y-only clearance is gated by accept_5y.
    Returns (status, window_used, avg_of_that_window)."""
    best_long = None
    for w in _long_windows(any_long_window):
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


def _na_reason(s_newest: List[Optional[float]], need: int, what: str) -> str:
    """Human-readable reason why a check is NA for this data series."""
    avail = sum(1 for v in s_newest if v is not None)
    if avail == 0:
        return (f"{what} not reported by the data source for this company — "
                "cannot evaluate; NA never disqualifies")
    return (f"only {avail} annual value(s) of {what} available — this test "
            f"needs ≥{need}; NA never disqualifies")


def _trend_distortion_idx(s_newest: List[Optional[float]],
                          any_long_window: bool) -> Optional[int]:
    """Temporary-distortion detector for trend checks.

    If the long-window durable-trend test FAILS but PASSES after removing
    exactly ONE annual value, that year is very likely a one-off distortion
    (M&A/integration charges, impairment, legal settlement, tax one-off,
    divestiture, 53rd-week effects). Returns the newest-first index of the
    distorting year, else None."""
    avail = sum(1 for v in s_newest if v is not None)
    for w in _long_windows(any_long_window):
        if avail < min(w, 8):
            continue
        win = _window(s_newest, w)
        if _consistently_increasing(win, min_points=8):
            return None  # not failing — nothing to explain
        for i in range(len(win)):
            if win[i] is None:
                continue
            trial = win[:i] + win[i + 1:]
            if _consistently_increasing(trial, min_points=8):
                return i
        return None
    return None


def _avg_distortion(s_newest: List[Optional[float]], threshold: float,
                    any_long_window: bool):
    """Temporary-distortion detector for average-threshold checks (ROE/ROIC).

    If the long-window average FAILS the threshold but clears it once the
    single WORST year is excluded — and that year is a clear outlier vs the
    rest — return (excluded_value, avg_without_it); else None."""
    for w in _long_windows(any_long_window):
        vals = [v for v in _window(s_newest, w) if v is not None]
        if len(vals) < min(w, 6) or len(vals) <= 5:
            continue
        avg = sum(vals) / len(vals)
        if avg >= threshold:
            return None  # not failing
        worst = min(vals)
        rest = [v for v in vals if v != worst] or vals
        avg_rest = sum(rest) / len(rest)
        # outlier test: worst year negative or far below the other years
        spread = (sum((v - avg_rest) ** 2 for v in rest) / len(rest)) ** 0.5
        is_outlier = worst < 0 or worst < avg_rest - 2 * max(spread, 1e-9)
        if avg_rest >= threshold and is_outlier:
            return worst, avg_rest
        return None
    return None


def _single_year_jump(s_newest: List[Optional[float]], n: int,
                      min_jump: float = 0.25) -> bool:
    """True if any single year-over-year increase within the last n years
    exceeds min_jump (e.g. +25%) — a step-change signature of M&A."""
    win = [v for v in _window(s_newest, n) if v is not None]
    s = list(reversed(win))
    return any(a > 0 and (b - a) / a >= min_jump for a, b in zip(s, s[1:]))


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
        if self.n_fail != 0 or self.applicable < 8:
            return False
        # "Growth-stage transition" / "10-12% band" WARNs on ROE/ROIC are
        # judgment flags we ourselves softened from FAIL because the
        # historical average is unrepresentative (e.g. TMO post-acquisition,
        # PANW newly profitable) — they neither break the core rule nor
        # consume the warn budget.
        def _soft(c):
            d = (c.detail or "")
            return c.status == WARN and ("Growth-stage" in d or "10-12%" in d)
        hard_warns = sum(1 for c in self.checks if c.status == WARN and not _soft(c))
        if hard_warns > 2:
            return False
        by_name = {c.name: c for c in self.checks}
        for n in self.CORE_CHECKS:
            c = by_name.get(n)
            if c is None:
                continue
            if c.status not in (PASS, NA) and not _soft(c):
                return False
        return True

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
               any_long_window: bool = False,
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

    _win_desc = ("ANY of 20/15/10y" if any_long_window
                 else "the full 20y window")

    # ---- 1. Sales consistently increasing — long-window trend (20y-only
    # by default; 20/15/10y any-pass when the toggle is on; a 5y-only
    # pass is WARN unless the accept-5y toggle allows it)
    rev = _first_present(_series(inc, "revenue"))
    st, w, cagr = _multi_window_trend(rev, accept_5y, any_long_window)
    res.metrics["rev_cagr_5y"] = _cagr(_window(rev, 5))
    res.metrics["rev_cagr_10y"] = _cagr(_window(rev, 10))
    res.metrics["rev_cagr_15y"] = _cagr(_window(rev, 15))
    if st == NA:
        add("Sales increasing (multi-window)", NA, None,
            _na_reason(rev, 5, "revenue history"))
    elif st == FAIL and _trend_distortion_idx(rev, any_long_window) is not None:
        add("Sales increasing (multi-window)", WARN,
            (_fmt_pct(cagr) + " CAGR") if cagr is not None else None,
            "Trend passes once a SINGLE distorted year is excluded — likely a "
            "temporary distortion (divestiture, M&A timing, FX, 53rd week). "
            "Review that year manually; underlying trend intact")
    else:
        add("Sales increasing (multi-window)", st,
            (_fmt_pct(cagr) + f" CAGR ({_wl(w)})") if cagr is not None else None,
            f"Durable-trend test over {_win_desc}"
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
    st_ni, w_ni, cagr_ni = _multi_window_trend(ni, accept_5y, any_long_window)
    if st_ni == NA:
        add("Net income increasing (multi-window)", NA, None,
            _na_reason(ni, 5, "net income history"))
    elif st_ni == FAIL:
        st_oi, w_oi, _c = _multi_window_trend(oi, accept_5y, any_long_window)
        _ni_dist = _trend_distortion_idx(ni, any_long_window)
        if st_oi == PASS:
            add("Net income increasing (multi-window)", WARN,
                (_fmt_pct(cagr_ni) + " CAGR") if cagr_ni is not None else None,
                "Net income choppy but OPERATING income consistently rising "
                f"({_wl(w_oi)}) — course-approved fallback (one-off items)")
        elif _ni_dist is not None:
            add("Net income increasing (multi-window)", WARN,
                (_fmt_pct(cagr_ni) + " CAGR") if cagr_ni is not None else None,
                "Trend passes once a SINGLE distorted year is excluded — "
                "likely a temporary one-off (M&A/integration charges, "
                "impairment, legal settlement, tax one-off). Review that "
                "year manually; underlying trend intact")
        elif _improving_transition(ni_w) or _improving_transition(oi_w):
            add("Net income increasing (multi-window)", WARN,
                (_fmt_pct(cagr_ni) + " CAGR") if cagr_ni is not None else None,
                "Profitability improving on a smoothed long-term basis but not "
                "yet meeting the mature-company consistency test (growth-stage)")
        else:
            add("Net income increasing (multi-window)", FAIL,
                (_fmt_pct(cagr_ni) + " CAGR") if cagr_ni is not None else None,
                "Neither net income nor operating income consistently rising "
                f"over {_win_desc} (or 5y)")
    else:
        add("Net income increasing (multi-window)", st_ni,
            (_fmt_pct(cagr_ni) + f" CAGR ({_wl(w_ni)})") if cagr_ni is not None else None,
            "5y-only pass — review flag" if st_ni == WARN else "")

    # ---- 3. CFO consistently increasing — multi-window
    ocf = _first_present(_series(cf, "ncfo"))
    st, w, cagr = _multi_window_trend(ocf, accept_5y, any_long_window)
    res.metrics["cfo_cagr_10y"] = _cagr(_window(ocf, 10))
    if st == NA:
        add("CFO increasing (multi-window)", NA, None,
            _na_reason(ocf, 5, "operating cash flow history"))
    elif st == FAIL and _trend_distortion_idx(ocf, any_long_window) is not None:
        add("CFO increasing (multi-window)", WARN,
            (_fmt_pct(cagr) + " CAGR") if cagr is not None else None,
            "Trend passes once a SINGLE distorted year is excluded — likely "
            "temporary (working-capital swing, M&A/litigation cash outflow, "
            "tax timing). Review that year manually; underlying trend intact")
    else:
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
        add("FCF positive", NA, None,
            "Neither FCF nor CFO+capex reported by the data source — cannot "
            "compute free cash flow; NA never disqualifies")
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
        _na_reason(gm_w, 4, "gross margin (needs COGS split)")
        if gm_status == NA else
        ("Positive but compressed vs 5y average — review whether temporary"
         if gm_status == WARN else ""))

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
        _na_reason(nm_w, 4, "net profit margin")
        if nm_status == NA else
        ("Positive but compressed / growth-transition — review whether temporary"
         if nm_status == WARN else ""))

    # ---- 7. ROE >= 12% — multi-window average (20/15/10y any-pass; 5y gated)
    roe = _series(rat, "roe")
    roe_latest = next((v for v in roe if v is not None), None)
    st, w, roe_avg = _multi_window_avg(roe, 12.0, accept_5y, any_long_window)
    if st == NA:
        add("ROE ≥ 12%", NA, None, _na_reason(roe, 6, "ROE history"))
    elif any(v is not None and v < 0
             for v in _window(_series(bal, "equity"), WINDOW_10Y)):
        add("ROE ≥ 12%", WARN, "negative equity in window",
            "ROE distorted by negative shareholder equity (often buybacks, "
            "e.g. MCD/YUM/AZO) — course explicitly says judge manually")
    else:
        val = f"{_wl(w)} avg {roe_avg:.1f}%" if roe_avg is not None else None
        if val and roe_latest is not None:
            val += f", latest {roe_latest:.1f}%"
        _roe_dist = _avg_distortion(roe, 12.0, any_long_window) if st == FAIL else None
        if st == FAIL and _roe_dist is not None:
            add("ROE ≥ 12%", WARN, val,
                f"Average clears 12% ({_roe_dist[1]:.1f}%) once ONE outlier "
                f"year ({_roe_dist[0]:.1f}%) is excluded — likely a temporary "
                "distortion (impairment/M&A charge, one-off loss, equity "
                "swing from buybacks). Review that year manually")
        elif st == FAIL and roe_avg is not None and roe_avg >= 10:
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
    st, w, roic_avg = _multi_window_avg(roic_computed, 12.0, accept_5y,
                                        any_long_window)
    if st == NA:
        _missing = [nm for nm, s_ in (("EBIT", ebit_s), ("pretax income", pretax_s),
                                      ("income tax", tax_s), ("equity", equity_s),
                                      ("LT debt", debt_s), ("cash", cash_s))
                    if not any(v is not None for v in s_)]
        add("ROIC ≥ 12%", NA, None,
            "Cannot compute EBIT x (1-tax) / (Equity+Debt-Cash): "
            + (f"source is missing {', '.join(_missing)}" if _missing
               else _na_reason(roic_computed, 6, "computable ROIC years"))
            + "; NA never disqualifies")
    else:
        val = f"{_wl(w)} avg {roic_avg:.1f}%" if roic_avg is not None else None
        if val and roic_latest is not None:
            val += f", latest {roic_latest:.1f}%"
        _roic_dist = _avg_distortion(roic_computed, 12.0, any_long_window) if st == FAIL else None
        if st == FAIL and _roic_dist is not None:
            add("ROIC ≥ 12%", WARN, val,
                f"Average clears 12% ({_roic_dist[1]:.1f}%) once ONE outlier "
                f"year ({_roic_dist[0]:.1f}%) is excluded — likely a temporary "
                "distortion (goodwill/impairment charge, M&A year with "
                "inflated invested capital before synergies). Review manually")
        elif st == FAIL and roic_avg is not None and roic_avg >= 10:
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
        add("Current ratio ≥ 1", NA, None,
            "Current assets/liabilities split not reported by the data source "
            "(common for insurers/utilities with unclassified balance sheets); "
            "NA never disqualifies")
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
        add("Debt/EBITDA ≤ 3", NA, None,
            ("EBITDA not reported/derivable from this source" if ebitda is None
             else "long-term debt not reported by this source")
            + " — cannot compute the ratio; NA never disqualifies")
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
            None if ocf_latest is None else "CFO negative",
            "Operating cash flow not reported — the ratio needs interest/CFO; "
            "NA never disqualifies" if ocf_latest is None else "")
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
        _rec_avail = sum(1 for v in rec if v is not None)
        add("Receivables ≤ sales growth", NA, None,
            ("Trade receivables not reported by the data source — many "
             "companies fold them into other current assets" if _rec_avail == 0
             else f"only {_rec_avail} overlapping receivables/revenue year(s) "
                  "— need ≥2 to compare growth rates")
            + "; NA never disqualifies")
    elif rec_cagr <= rev_cagr + 0.03:  # 3pp tolerance
        add("Receivables ≤ sales growth", PASS,
            f"recv {_fmt_pct(rec_cagr)} vs sales {_fmt_pct(rev_cagr)}")
    elif _single_year_jump(rec, WINDOW_10Y) and not _single_year_jump(rev, WINDOW_10Y, 0.20):
        add("Receivables ≤ sales growth", WARN,
            f"recv {_fmt_pct(rec_cagr)} vs sales {_fmt_pct(rev_cagr)}",
            "Excess receivables growth traces to a SINGLE step-change year — "
            "signature of an acquisition adding acquired receivables, not "
            "channel stuffing. Verify the M&A year, then judge organically")
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
            "finviz has no analyst EPS estimates for this ticker (thin "
            "coverage / foreign listing) — does not count against is_great")

    # ---------------- Intrinsic value via DCF (informational, no check) ----
    # StockOracle DCF-20yr replica, calibration v13 (see vmi/dcf_v13.py).
    # DCF STRUCTURE verified to the cent against the Visa calculator
    # screenshot in Lesson 5 (p.6). Growth + base-flow choice fitted on 36
    # large caps vs StockOracle "Base IV": 36/36 within ±7%, 32/36 within
    # ±5% (scanner/calib/blend_fit_v13.py). Base flow is a continuous
    # per-sector mix over annual SEC flows, TTM flows (stockanalysis.com)
    # and 3y averages; growth is a deterministic blend of analyst estimates
    # + fundamentals with sector terms. No invented caps or minimums.
    price = g.get("price")
    shares = g.get("shares_outstanding")
    beta = g.get("beta")
    res.metrics["price"] = price

    def _latest_bal(key):
        s = _series(bal, key)
        return next((v for v in s if v is not None), None) if s else None

    ety = g.get("eps_this_y")
    eny = proj1
    g5 = proj5

    _iv = None
    if shares:
        capex_s = _series(cf, "capex")
        # TTM flows from stockanalysis.com (cached HTTP; graceful fallback
        # to annual flows inside compute_iv when unavailable)
        ttm = None
        try:
            from .stockanalysis import fetch_statement as _sa_fetch
            _cfT = _sa_fetch(res.ticker, "cashflow")
            if _cfT and _cfT.get("datekey") and _cfT["datekey"][0] == "TTM":
                ttm = {"ocf": (_cfT.get("ncfo") or [None])[0],
                       "capex": (_cfT.get("capex") or [None])[0],
                       "ni": (_cfT.get("cash_flow_statement_net_income")
                              or [None])[0]}
        except Exception:
            ttm = None
        try:
            from .dcf_v13 import compute_iv as _compute_iv
            _iv = _compute_iv(
                sector=res.sector, industry=res.industry,
                shares=shares, beta=beta, g5=g5, eny=eny, ety=ety,
                fwd_eps=g.get("fwd_eps"),
                ocf_series=ocf or [], capex_series=capex_s or [],
                ni_series=ni or [], rev_series=rev or [],
                cash=_latest_bal("cash") or 0,
                sti=_latest_bal("shortTermInvestments") or 0,
                std=_latest_bal("shortTermDebt") or 0,
                ltd=_latest_bal("longTermDebt") or 0,
                ttm=ttm)
        except Exception:
            _iv = None

    if _iv is not None:
        iv_ps = _iv["iv_ps"]
        res.metrics["intrinsic_value"] = round(iv_ps, 2)
        res.metrics["dcf_growth_used"] = round(_iv["g_pct"], 1)
        res.metrics["dcf_base_flow"] = _iv["base_desc"]
        res.metrics["dcf_sector_group"] = _iv["sector_group"]
        res.metrics["dcf_discount_rate"] = round(_iv["disc_pct"], 2)
        if price and iv_ps > 0:
            # Positive = trading below IV (discount); negative = premium.
            res.metrics["discount_pct"] = round((iv_ps - price) / iv_ps * 100, 1)
        else:
            res.metrics["discount_pct"] = None
    else:
        res.metrics["intrinsic_value"] = None
        res.metrics["discount_pct"] = None

    # ---------------- DIRECT DCF (no calibrated blend) -------------------
    # Growth = AVERAGE of providers per the StockOracle recipe (GuruFocus
    # + Finviz + Zacks + stockanalysis consensus; GuruFocus/Zacks are
    # currently 403-blocked server-side but auto-join the average if they
    # unblock — see scanner/vmi/growth.py). Bands: yrs 1-5 at g, yrs 6-10
    # at min(g, 15%), yrs 11-20 at 4%. TTM EPS base.
    try:
        from .growth import projected_growth as _proj_g
        _g_avg, _g_src = _proj_g(res.ticker, g5)
    except Exception:
        _g_avg, _g_src = g5, "finviz"
    res.metrics["direct_growth_used"] = (
        round(_g_avg, 2) if _g_avg is not None else None)
    res.metrics["direct_growth_sources"] = _g_src
    try:
        from .dcf_v13 import compute_iv_direct as _civ_d
        _ttm_ni = (ttm or {}).get("ni") if shares else None
        _ivd = _civ_d(shares=shares, beta=beta, g5=_g_avg,
                      ni_series=ni or [],
                      cash=_latest_bal("cash") or 0,
                      sti=_latest_bal("shortTermInvestments") or 0,
                      std=_latest_bal("shortTermDebt") or 0,
                      ltd=_latest_bal("longTermDebt") or 0,
                      ttm_ni=_ttm_ni) if shares else None
    except Exception:
        _ivd = None
    if _ivd is not None:
        _ivd_ps = _ivd["iv_ps"]
        res.metrics["intrinsic_value_direct"] = round(_ivd_ps, 2)
        if price and _ivd_ps > 0:
            res.metrics["discount_pct_direct"] = round(
                (_ivd_ps - price) / _ivd_ps * 100, 1)
        else:
            res.metrics["discount_pct_direct"] = None
    else:
        res.metrics["intrinsic_value_direct"] = None
        res.metrics["discount_pct_direct"] = None

    # ---------------- TTM statistics (S&P Global via stockanalysis.com) --
    # Same data family StockOracle displays: MSFT ROE 34.04 exact match,
    # PE 27.68 vs 27.69, FCF yield 1.82 vs 1.81. Stored as ttm_* metrics.
    _st = {}
    try:
        from .stockanalysis import fetch_statistics as _sa_stats
        _st = _sa_stats(res.ticker) or {}
    except Exception:
        _st = {}

    # FALLBACK CHAIN so stats are never NA just because one site failed:
    #   1. stockanalysis.com /statistics  (S&P Global — preferred)
    #   2. Yahoo Finance (yfinance .info) — market-data driven
    #   3. Computed from SEC XBRL filings already in hand
    # Every fallback value is still a published or arithmetic-derived
    # figure — nothing invented.
    def _fill(key, val):
        if _st.get(key) is None and val is not None:
            try:
                v = float(val)
            except (TypeError, ValueError):
                return
            if v == v:  # not NaN
                _st[key] = round(v, 3)

    if any(_st.get(k) is None for k in (
            "roe", "roic", "pe", "fwd_pe", "peg", "div_yield", "fcf_yield",
            "current_ratio", "debt_equity", "z_score",
            "interest_coverage", "debt_ebitda")):
        # ---- Tier 2: Yahoo Finance ----
        try:
            import yfinance as _yf
            _yi = _yf.Ticker(res.ticker).info or {}
            _fill("roe", (_yi.get("returnOnEquity") or 0) * 100
                  if _yi.get("returnOnEquity") is not None else None)
            _fill("roa", (_yi.get("returnOnAssets") or 0) * 100
                  if _yi.get("returnOnAssets") is not None else None)
            _fill("pe", _yi.get("trailingPE"))
            _fill("fwd_pe", _yi.get("forwardPE"))
            _fill("peg", _yi.get("trailingPegRatio"))
            _fill("div_yield", _yi.get("dividendYield"))
            _fill("current_ratio", _yi.get("currentRatio"))
            _fill("quick_ratio", _yi.get("quickRatio"))
            _fill("debt_equity", (_yi.get("debtToEquity") or 0) / 100
                  if _yi.get("debtToEquity") is not None else None)
            _fill("profit_margin", (_yi.get("profitMargins") or 0) * 100
                  if _yi.get("profitMargins") is not None else None)
            _fill("operating_margin", (_yi.get("operatingMargins") or 0)
                  * 100 if _yi.get("operatingMargins") is not None else None)
            _fill("gross_margin", (_yi.get("grossMargins") or 0) * 100
                  if _yi.get("grossMargins") is not None else None)
            if (_yi.get("freeCashflow") and _yi.get("marketCap")):
                _fill("fcf_yield",
                      _yi["freeCashflow"] / _yi["marketCap"] * 100)
        except Exception:
            pass
        # ---- Tier 3: computed from SEC XBRL already fetched ----
        try:
            _ni0 = next((v for v in (ni or []) if v is not None), None)
            _eq0 = _latest_bal("equity")
            if _ni0 and _eq0:
                _fill("roe", _ni0 / _eq0 * 100)
            _p_ttm = (ttm or {}).get("ni") or _ni0
            if price and shares and _p_ttm and _p_ttm > 0:
                _fill("pe", price * shares / _p_ttm)
            _ltd0 = _latest_bal("longTermDebt") or 0
            _std0 = _latest_bal("shortTermDebt") or 0
            _ebitda_s = _series(inc, "ebitda")
            _ebitda0 = next((v for v in (_ebitda_s or [])
                             if v is not None), None)
            if _ebitda0 and _ebitda0 > 0:
                _fill("debt_ebitda", (_ltd0 + _std0) / _ebitda0)
            _ie_s = _series(inc, "income_statement_interest_expense")
            _ie0 = next((v for v in (_ie_s or []) if v), None)
            _ebit_s = _series(inc, "ebit")
            _ebit0 = next((v for v in (_ebit_s or [])
                           if v is not None), None)
            if _ebit0 and _ie0:
                _fill("interest_coverage", abs(_ebit0 / _ie0))
            if _eq0:
                _fill("debt_equity", (_ltd0 + _std0) / _eq0)
            _ocf0 = next((v for v in (ocf or []) if v is not None), None)
            _cap_s = _series(cf, "capex")
            _cap0 = next((v for v in (_cap_s or []) if v is not None), 0)
            if _ocf0 and price and shares:
                _fill("fcf_yield",
                      (_ocf0 - (_cap0 or 0)) / (price * shares) * 100)
        except Exception:
            pass

    for _k in ("roe", "roic", "roa", "pe", "fwd_pe", "peg", "fcf_yield",
               "div_yield", "current_ratio", "debt_equity", "debt_ebitda",
               "interest_coverage", "z_score", "f_score", "eps_growth_3y"):
        if _st.get(_k) is not None:
            res.metrics[f"ttm_{_k}"] = _st[_k]

    # ---------------- Sort scores (0-100, equal weight, no fitted params) -
    # Each component is a documented public metric capped at a stated
    # full-marks level, then averaged. No invented weights.
    def _cap01(v, cap):
        if v is None:
            return None
        return max(0.0, min(1.0, v / cap))

    def _avgp(parts):
        ps = [p for p in parts if p is not None]
        return round(sum(ps) / len(ps) * 100, 1) if ps else None

    def _updown(series):
        s = [v for v in _oldest_first(series) if v is not None]
        if len(s) < 4:
            return None
        ups = sum(1 for a, b in zip(s, s[1:]) if b > a)
        return ups / (len(s) - 1)

    # Financial strength: Altman Z (10=full), interest coverage (50=full),
    # current ratio (3=full), low debt/EBITDA (0=full, >=5 = zero).
    _de = _st.get("debt_ebitda")
    res.metrics["sort_financial_strength"] = _avgp([
        _cap01(_st.get("z_score"), 10),
        _cap01(_st.get("interest_coverage"), 50),
        _cap01(_st.get("current_ratio"), 3),
        (5 - min(_de, 5)) / 5 if _de is not None else None])

    # Predictability: fraction of positive YoY changes in revenue and NI.
    res.metrics["sort_predictability"] = _avgp([
        _updown(rev), _updown(ni)])

    # Profitability (ROE + ROIC packaged in, per request): ROE (50=full),
    # ROIC (50=full), net margin (40=full) — S&P TTM values.
    res.metrics["sort_profitability"] = _avgp([
        _cap01(_st.get("roe"), 50),
        _cap01(_st.get("roic"), 50),
        _cap01(_st.get("profit_margin"), 40)])

    # Growth: 5y revenue CAGR, 5y NI CAGR (stored as fractions -> x100),
    # analyst 3-5y EPS growth. 40%/yr = full marks each.
    _rc5 = res.metrics.get("rev_cagr_5y")
    _nc5 = res.metrics.get("ni_cagr_5y")
    res.metrics["sort_growth"] = _avgp([
        _cap01(_rc5 * 100, 40) if _rc5 is not None else None,
        _cap01(_nc5 * 100, 40) if _nc5 is not None else None,
        _cap01(g5, 40)])

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
