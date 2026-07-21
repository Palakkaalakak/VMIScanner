"""Dow Jones Industrial Average universe source — Wikipedia constituents.

Added per user instruction ("expand the scanner's universe to include
the Dow Jones").  Same seam as sp500.py: returns a list of dicts with
{ticker, company, sector, sub_industry, headquarters, industry, country,
market_cap}.  Nearly all Dow 30 members are also S&P 500 members, so the
orchestrator merges by ticker (union, no duplicates) and tags each row
with its index membership for provenance.

The DJIA table on Wikipedia has no GICS sector column; the industry
free-text is used for both `sector` and `industry` so classify()'s
REIT/financial exclusion still works on the merged rows.
"""
import html
import re
from typing import Dict, List

from .http import get

WIKI_URL = ("https://en.wikipedia.org/api/rest_v1/page/html/"
            "Dow_Jones_Industrial_Average")


def _clean(cell: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", cell)).strip()


def fetch_dowjones(use_cache: bool = True,
                   cache_max_age: float = 86400 * 7) -> List[Dict]:
    """Scrape the current DJIA component table from Wikipedia."""
    page = get(WIKI_URL, use_cache=use_cache, cache_max_age=cache_max_age,
               domain_hint="wikipedia")
    i = page.find('id="constituents"')
    if i == -1:
        # fallback: the components table caption/anchor name has varied
        i = page.find("Components")
    if i == -1:
        raise RuntimeError("DJIA components table not found "
                           "(page layout changed?)")
    j = page.find("</table>", i)
    table_html = page[i:j]

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.S)
    out: List[Dict] = []
    for row in rows:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S)
        if len(cells) < 3:
            continue
        # DJIA table columns: Company | Exchange | Symbol | Industry | ...
        # find the symbol cell: first cell matching a ticker pattern that
        # is not the exchange name
        company = _clean(cells[0])
        ticker = ""
        industry = ""
        for k, c in enumerate(cells[1:], start=1):
            v = _clean(c)
            if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,6}", v) \
                    and v not in ("NYSE", "NASDAQ"):
                ticker = v
                if k + 1 < len(cells):
                    industry = _clean(cells[k + 1])
                break
        if not ticker or not company:
            continue
        out.append({
            "ticker": ticker.replace(".", "-"),
            "company": company,
            "sector": industry,       # no GICS column on DJIA table
            "sub_industry": industry,
            "headquarters": "",
            "industry": industry,
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
        print(r["ticker"], "-", r["company"], "-", r["industry"])
