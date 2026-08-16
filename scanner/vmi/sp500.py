"""S&P 500 universe source — scraped from Wikipedia's constituents table.

This is the PRIMARY universe for the scanner (per user requirement: scan
ALL S&P 500 companies, not just an arbitrary Finviz-filtered subset).

Design note — "largest filters first" infrastructure:
  This module is intentionally decoupled from any specific index. The
  universe-source functions here are the seam future universe sources
  should implement (e.g. `russell3000.py`, `all_us_listed.py`,
  `all_global_listed.py`) so the orchestrator (`scan.py`) can eventually
  chain multiple universes with the *cheapest / broadest filters first*:
    1. Index membership (S&P500 today; Russell 3000 / all-listed later)
    2. Free-text sector/industry exclusion (REIT/bank/financial — cheap,
       no network call, done on the metadata already in the universe row)
    3. Coarse numeric pre-filters (Finviz screener — one HTTP call for
       hundreds of tickers) — OPTIONAL, only when scanning very large
       universes where deep-checking everything is too slow/rate-limited
    4. Deep per-ticker fundamental checks (macrotrends/stockanalysis) —
       most expensive step, run last, only on what survives 1-3.
  For the S&P500-only case (today's deliverable) step 3 is skipped
  entirely since 503 tickers is small enough to deep-check directly.
"""
import html
import re
from typing import Dict, List

from .http import get

# The plain article URL (en.wikipedia.org/wiki/...) 403s our requests
# session (bot-detection on that path specifically); the REST API's
# rendered-HTML endpoint serves the identical table without issue.
WIKI_URL = "https://en.wikipedia.org/api/rest_v1/page/html/List_of_S%26P_500_companies"

# FALLBACK universe source (added 2026-08-16 after Wikipedia started
# returning HTTP 429 to the sandbox IP): the `datasets/s-and-p-500-companies`
# GitHub dataset is itself scraped from the SAME Wikipedia constituents
# table, exposes the SAME GICS Sector / Sub-Industry columns, and is
# served from raw.githubusercontent.com which does not rate-limit us.
CSV_FALLBACK_URL = ("https://raw.githubusercontent.com/datasets/"
                    "s-and-p-500-companies/main/data/constituents.csv")


def _fetch_sp500_csv_fallback() -> List[Dict]:
    """Parse the GitHub constituents CSV into the same row shape as
    the Wikipedia scraper. Used only when the Wikipedia fetch fails."""
    import csv
    import io
    text = get(CSV_FALLBACK_URL, use_cache=True, cache_max_age=86400 * 7,
               domain_hint="github")
    out: List[Dict] = []
    for r in csv.DictReader(io.StringIO(text)):
        ticker = (r.get("Symbol") or "").strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,6}", ticker):
            continue
        sub = (r.get("GICS Sub-Industry") or "").strip()
        out.append({
            "ticker": ticker.replace(".", "-"),
            "company": (r.get("Security") or "").strip(),
            "sector": (r.get("GICS Sector") or "").strip(),
            "sub_industry": sub,
            "headquarters": (r.get("Headquarters Location") or "").strip(),
            "industry": sub,  # alias for classify(), same as wiki path
            "country": "USA",
            "market_cap": "",
        })
    if len(out) < 400:
        raise RuntimeError(f"CSV fallback looks broken: only {len(out)} rows")
    return out


def _clean(cell: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", cell)).strip()


def fetch_sp500(use_cache: bool = True, cache_max_age: float = 86400 * 7) -> List[Dict]:
    """Scrape the current S&P 500 constituent table from Wikipedia.

    Returns a list of dicts: {ticker, company, sector, sub_industry,
    headquarters, industry, country, market_cap}. `sector` here is the
    GICS sector name (e.g. "Financials", "Real Estate", "Industrials")
    which is a cleaner exclusion signal than Finviz's free-text industry
    string.
    """
    try:
        html = get(WIKI_URL, use_cache=use_cache, cache_max_age=cache_max_age,
                   domain_hint="wikipedia")
    except Exception as e:
        # Wikipedia rate-limits (429) or blocks this IP sometimes — fall
        # back to the GitHub CSV mirror of the SAME constituents table.
        print(f"  Wikipedia universe fetch failed ({e}); using GitHub CSV fallback")
        return _fetch_sp500_csv_fallback()
    i = html.find('id="constituents"')
    if i == -1:
        raise RuntimeError("Wikipedia constituents table not found (page layout changed?)")
    j = html.find("</table>", i)
    table_html = html[i:j]

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.S)
    out: List[Dict] = []
    for row in rows:
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S)
        if len(cells) < 4:
            continue
        ticker = _clean(cells[0])
        if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,6}", ticker):
            continue  # skip header row / malformed rows
        out.append({
            "ticker": ticker.replace(".", "-"),  # BRK.B -> BRK-B (stockanalysis/macrotrends style)
            "company": _clean(cells[1]),
            "sector": _clean(cells[2]),
            "sub_industry": _clean(cells[3]) if len(cells) > 3 else "",
            "headquarters": _clean(cells[4]) if len(cells) > 4 else "",
            "industry": _clean(cells[3]) if len(cells) > 3 else "",  # alias for classify()
            "country": "USA",
            "market_cap": "",
        })
    return out


if __name__ == "__main__":
    rows = fetch_sp500(use_cache=False)
    print(f"{len(rows)} S&P 500 constituents")
    for r in rows[:5]:
        print(r)
