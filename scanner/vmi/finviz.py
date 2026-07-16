"""Finviz screener scraping — the universe pre-filter.

Implements the VMI course's recommended Finviz screen (from Lesson 9 /
Quick Reference J2), used ONLY as a coarse pre-filter to shrink the
universe before deep per-ticker fundamental checks:

  - Sales growth past 5 years: positive
  - EPS growth past 5 years: positive
  - EPS growth this year: positive
  - EPS growth next year: positive
  - EPS growth next 5 years: positive   (=> "positive projected growth rate")
  - ROE > 10%   (final check tightens to >= 12-15%)
  - Current ratio > 1

PEG < 2 from the course screen is deliberately EXCLUDED here because the
user asked for business quality only, no valuation.
"""
import re
from typing import Dict, List

from .http import get

BASE = "https://finviz.com/screener.ashx"

FILTERS = ",".join([
    "fa_curratio_o1",      # Current ratio > 1
    "fa_eps5years_pos",    # EPS growth past 5 years positive
    "fa_epsyoy_pos",       # EPS growth this year positive
    "fa_epsyoy1_pos",      # EPS growth next year positive
    "fa_estltgrowth_pos",  # EPS growth next 5 years positive
    "fa_roe_o10",          # ROE > 10%
    "fa_sales5years_pos",  # Sales growth past 5 years positive
])


def _total_count(html: str) -> int:
    m = re.search(r"#\d+\s*/\s*(\d+)", html)
    return int(m.group(1)) if m else 0


def _parse_rows(html: str) -> List[Dict]:
    """Parse screener table rows (v=111 overview view)."""
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        if len(tds) < 10:
            continue

        def clean(x: str) -> str:
            return re.sub(r"<[^>]+>", "", x).strip()

        ticker = clean(tds[1])
        if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,7}", ticker):
            continue
        rows.append({
            "ticker": ticker,
            "company": clean(tds[2]),
            "sector": clean(tds[3]),
            "industry": clean(tds[4]),
            "country": clean(tds[5]),
            "market_cap": clean(tds[6]),
        })
    return rows


def screen_universe(use_cache: bool = True, cache_max_age: float = 86400) -> List[Dict]:
    """Run the finviz pre-filter, paginate, return list of candidate dicts."""
    out: List[Dict] = []
    seen = set()
    r = 1
    total = None
    while True:
        url = f"{BASE}?v=111&f={FILTERS}&r={r}"
        html = get(url, use_cache=use_cache, cache_max_age=cache_max_age)
        if total is None:
            total = _total_count(html)
        rows = _parse_rows(html)
        new = [x for x in rows if x["ticker"] not in seen]
        if not new:
            break
        for x in new:
            seen.add(x["ticker"])
            out.append(x)
        r += 20
        if total and r > total:
            break
    return out


# ---------------------------------------------------------------- estimates
# Custom view v=152 column ids (verified against live header row):
#   1=Ticker, 8=Forward P/E, 17=EPS This Y, 18=EPS Next Y, 19=EPS Past 5Y,
#   20=EPS Next 5Y, 24=Shares Outstanding, 48=Beta, 65=Price
_EST_COLS = "0,1,8,17,18,19,20,24,48,65"


def _parse_est_rows(html: str) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for tr in re.findall(r"<tr[^>]*valign=\"top\"[^>]*>(.*?)</tr>", html, re.S):
        raw_tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        tds = [re.sub(r"<[^>]+>", "", t).strip() for t in raw_tds]
        if len(tds) < 10:
            continue
        # The ticker cell renders its text twice (visible + styled copy) so
        # stripped text comes out doubled ("AAAPL"); the link href is the
        # reliable source: stock?t=AAPL or quote.ashx?t=AAPL.
        m = re.search(r"(?:quote\.ashx|stock)\?t=([A-Za-z0-9.\-]+)", raw_tds[1])
        if not m:
            continue
        ticker = m.group(1).upper()

        def pct(x: str):
            x = x.replace("%", "").strip()
            try:
                return float(x)
            except ValueError:
                return None

        def num(x: str):
            """Parse plain numbers and 15.20B / 890.5M style abbreviations."""
            x = x.replace(",", "").strip()
            mult = 1.0
            if x[-1:] in ("B", "b"):
                mult, x = 1e9, x[:-1]
            elif x[-1:] in ("M", "m"):
                mult, x = 1e6, x[:-1]
            elif x[-1:] in ("K", "k"):
                mult, x = 1e3, x[:-1]
            try:
                return float(x) * mult
            except ValueError:
                return None

        fwd_pe = num(tds[2])
        price = num(tds[9])
        out[ticker] = {
            "eps_this_y": pct(tds[3]), "eps_next_y": pct(tds[4]),
            "eps_past_5y": pct(tds[5]), "eps_next_5y": pct(tds[6]),
            "shares_outstanding": num(tds[7]), "beta": num(tds[8]),
            "price": price,
            # Forward EPS ($) derived from Forward P/E — the IV model's base flow.
            "fwd_eps": (price / fwd_pe) if (price and fwd_pe and fwd_pe > 0) else None,
        }
    return out


def fetch_growth_estimates(tickers: List[str], use_cache: bool = True,
                           cache_max_age: float = 86400) -> Dict[str, Dict]:
    """Bulk analyst growth estimates via finviz custom view — paginates the
    whole S&P500 index screen (~25 pages of 20) rather than one URL per
    ticker. Returns {ticker: {eps_this_y, eps_next_y, eps_past_5y,
    eps_next_5y}} in percent (12.5 = +12.5%/yr); unknown tickers -> None."""
    got: Dict[str, Dict] = {}
    r = 1
    total = None
    while True:
        url = f"{BASE}?v=152&f=idx_sp500&c={_EST_COLS}&r={r}"
        html = get(url, use_cache=use_cache, cache_max_age=cache_max_age)
        if total is None:
            total = _total_count(html)
        rows = _parse_est_rows(html)
        if not rows:
            break
        before = len(got)
        got.update(rows)
        if len(got) == before:
            break
        r += 20
        if total and r > total:
            break
    out: Dict[str, Dict] = {}
    for t in tickers:
        k = t.upper().replace(".", "-")
        out[t] = got.get(k) or got.get(k.replace("-", "."))
    return out
