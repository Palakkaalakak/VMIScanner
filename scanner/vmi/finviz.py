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
