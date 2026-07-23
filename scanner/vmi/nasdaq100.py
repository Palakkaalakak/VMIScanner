"""Nasdaq-100 universe source — stockanalysis.com constituent list.

Added per user instruction ("expand to include their universes"):
catches large non-S&P500 names such as MELI and foreign primary
listings like ASML. Same seam as sp500.py — returns a list of dicts
{ticker, company, sector, industry, ...} for the orchestrator to merge.

Source note: the Wikipedia Nasdaq-100 article no longer carries a
components table in its REST-rendered HTML (verified 2026-07), so we
use stockanalysis.com's list page — the same site already trusted as a
per-ticker fundamentals source elsewhere in this package
(stockanalysis.py). Sector/industry come back empty here; that only
means classify() can't pre-exclude these rows, which is safe: the
deep-check step judges them on fundamentals regardless, and NDX has no
REITs/banks of consequence.
"""
import re
from typing import Dict, List

from .http import get

LIST_URL = "https://stockanalysis.com/list/nasdaq-100-stocks/"


def fetch_nasdaq100(use_cache: bool = True,
                    cache_max_age: float = 86400 * 7) -> List[Dict]:
    """Fetch current Nasdaq-100 constituents from stockanalysis.com."""
    page = get(LIST_URL, use_cache=use_cache, cache_max_age=cache_max_age)
    # Rows render as <a href="/stocks/meli/">MELI</a></td><td>Company.
    pairs = re.findall(
        r'href="/stocks/([a-z0-9.\-]+)/"[^>]*>[^<]*</a>\s*</td>\s*<td[^>]*>([^<]*)',
        page)
    out: List[Dict] = []
    seen = set()
    if pairs and len(pairs) >= 50:
        for slug, company in pairs:
            t = slug.upper().replace(".", "-")
            if t in seen:
                continue
            seen.add(t)
            out.append({"ticker": t, "company": company.strip(),
                        "sector": "", "sub_industry": "", "industry": "",
                        "country": "", "market_cap": ""})
    else:
        # Fallback: ticker links only, no company names.
        for slug in re.findall(r'href="/stocks/([a-z0-9.\-]+)/"', page):
            t = slug.upper().replace(".", "-")
            if t in seen:
                continue
            seen.add(t)
            out.append({"ticker": t, "company": "", "sector": "",
                        "sub_industry": "", "industry": "",
                        "country": "", "market_cap": ""})
    if len(out) < 90:
        raise RuntimeError(
            f"Nasdaq-100 parse suspicious: only {len(out)} rows")
    return out


if __name__ == "__main__":
    rows = fetch_nasdaq100(use_cache=False)
    print(f"{len(rows)} Nasdaq-100 constituents")
    for r in rows[:5]:
        print(r)
