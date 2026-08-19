"""Dow Jones Industrial Average universe source — stockanalysis.com list.

Added per user instruction ("expand the scanner's universe to include
the Dow Jones").  Same seam as sp500.py: returns a list of dicts with
{ticker, company, sector, sub_industry, headquarters, industry, country,
market_cap}.  Nearly all Dow 30 members are also S&P 500 members, so the
orchestrator merges by ticker (union, no duplicates) and tags each row
with its index membership for provenance.

Source note: the Wikipedia DJIA article no longer carries a components
table in its REST-rendered HTML (verified 2026-08 — components now
appear only in a navbox <ul> with company names and no tickers), so we
use stockanalysis.com's list page — the same site already trusted for
the Nasdaq-100 list (nasdaq100.py) and as a per-ticker fundamentals
source (stockanalysis.py).  Sector/industry come back empty here; that
is safe because the Dow 30 are almost all S&P 500 members whose rows
carry GICS sectors, and the merge in scan.py is by ticker union.
"""
import re
from typing import Dict, List

from .http import get

LIST_URL = "https://stockanalysis.com/list/dow-jones-stocks/"


def fetch_dowjones(use_cache: bool = True,
                   cache_max_age: float = 86400 * 7) -> List[Dict]:
    """Fetch current DJIA constituents from stockanalysis.com."""
    page = get(LIST_URL, use_cache=use_cache, cache_max_age=cache_max_age)
    # Scope to the constituents table body — page nav also contains
    # /stocks/... links (SCREENER, COMPARE, ...) we must not pick up.
    tbl = re.search(r"<tbody[^>]*>(.*?)</tbody>", page, re.S)
    if not tbl:
        raise RuntimeError("DJIA list table not found "
                           "(page layout changed?)")
    body = tbl.group(1)
    out: List[Dict] = []
    seen = set()
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S):
        m = re.search(r'href="/stocks/([a-z0-9.\-]+)/"[^>]*>', row)
        if not m:
            continue
        t = m.group(1).upper().replace(".", "-")
        if t in seen:
            continue
        seen.add(t)
        tds = [re.sub(r"<[^>]+>", "", x).strip()
               for x in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        # Layout: rank | ticker | company | ... — company is the first
        # cell after the ticker cell that is non-numeric text.
        company = ""
        for cell in tds:
            if cell and cell.upper() != t and not re.match(
                    r"^[\d$%,.\-+ ]+$", cell):
                company = cell
                break
        out.append({
            "ticker": t,
            "company": company,
            "sector": "",        # no GICS column on the list page; the
            "sub_industry": "",  # S&P500 merge supplies sectors anyway
            "headquarters": "",
            "industry": "",
            "country": "USA",
            "market_cap": "",
        })
    if len(out) < 25:
        raise RuntimeError(f"DJIA scrape returned only {len(out)} rows "
                           "(expected ~30) — layout changed?")
    return out


if __name__ == "__main__":
    rows = fetch_dowjones(use_cache=False)
    print(f"{len(rows)} DJIA components")
    for r in rows:
        print(r["ticker"], "-", r["company"])
