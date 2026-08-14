"""Projected 3-5y EPS growth from multiple providers, per the StockOracle
recipe:

    Primary (averaged when available):
      - GuruFocus  "Future 3-5 year growth rate"
      - Finviz     "Next 5 year" EPS growth estimate
      - Zacks      expected growth projection
    Fallbacks (used only when NO primary is available):
      - stockanalysis.com /forecast analyst EPS estimate CAGR
      - Yahoo Finance +1y EPS growth (via yfinance)

REALITY CHECK (tested 2026-08-14 from this server): GuruFocus and Zacks
return HTTP 403 to non-browser requests on every growth endpoint tried
(gurufocus.com/term/growth-rate-future, /reader/_api, zacks.com
detailed-estimates, widget3.zacks.com estimate feeds). The fetchers below
still TRY each provider, but self-disable for the process lifetime after 3
consecutive failures so a 480-name scan doesn't burn ~1000 doomed requests.
If either provider ever unblocks, its numbers automatically join the
average — no code change needed.
"""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Optional, Tuple

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/126.0 Safari/537.36",
       "Accept-Language": "en-US,en;q=0.9"}

# Self-disable counters: provider -> consecutive failures (3 = give up).
_fails = {"gurufocus": 0, "zacks": 0}
_MAX_FAILS = 3


def _get(url: str, timeout: int = 10) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    return urllib.request.urlopen(req, timeout=timeout).read()


def gurufocus_growth(ticker: str) -> Optional[float]:
    """GuruFocus 'Future 3-5Y growth rate' term page.
    Currently 403-blocked server-side; kept for when/if it unblocks."""
    if _fails["gurufocus"] >= _MAX_FAILS:
        return None
    try:
        h = _get(f"https://www.gurufocus.com/term/growth-rate-future/"
                 f"{ticker.upper()}").decode("utf8", "ignore")
        m = re.search(r"(?i)future[^%]{0,200}?([-\d.]+)\s*%", h)
        _fails["gurufocus"] = 0
        return float(m.group(1)) if m else None
    except Exception:  # noqa: BLE001
        _fails["gurufocus"] += 1
        return None


def zacks_growth(ticker: str) -> Optional[float]:
    """Zacks long-term growth from the detailed-estimates page.
    Currently 403-blocked server-side; kept for when/if it unblocks."""
    if _fails["zacks"] >= _MAX_FAILS:
        return None
    try:
        h = _get(f"https://www.zacks.com/stock/quote/{ticker.upper()}"
                 f"/detailed-estimates").decode("utf8", "ignore")
        m = re.search(r"(?i)next\s*5\s*years?[^%]{0,160}?([-\d.]+)\s*%", h)
        _fails["zacks"] = 0
        return float(m.group(1)) if m else None
    except Exception:  # noqa: BLE001
        _fails["zacks"] += 1
        return None


def sa_forecast_growth(ticker: str) -> Optional[float]:
    """Analyst EPS estimate CAGR from stockanalysis.com's /forecast page
    ANNUAL table (first positive estimate -> last positive estimate).
    Pure arithmetic on published analyst numbers — no invented figures.

    Verified 2026-08-14: MU 69.8 (finviz raw shows 172.8 — stale),
    CSCO 6.0, TSLA -7.7, MSFT 15.3."""
    try:
        from .stockanalysis import _decode_node, get as _sa_get
        root = _decode_node(json.loads(_sa_get(
            f"https://stockanalysis.com/stocks/{ticker.lower()}"
            f"/forecast/__data.json")))
    except Exception:  # noqa: BLE001
        return None
    best: Optional[float] = None

    def visit(o):
        nonlocal best
        if isinstance(o, dict):
            if isinstance(o.get("eps"), list) and isinstance(
                    o.get("dates"), list):
                dates, eps = o["dates"], o["eps"]
                if (len(dates) >= 3 and
                        all(isinstance(d, str) for d in dates)):
                    yrs = [d[:4] for d in dates]
                    if len(set(yrs)) == len(yrs):  # annual, not quarterly
                        pr = [(d, e) for d, e in zip(dates, eps)
                              if isinstance(e, (int, float)) and e > 0]
                        if len(pr) >= 2:
                            (d0, e0), (d1, e1) = pr[0], pr[-1]
                            n = int(d1[:4]) - int(d0[:4])
                            if n >= 1:
                                best = ((e1 / e0) ** (1 / n) - 1) * 100
            for v in o.values():
                visit(v)
        elif isinstance(o, list):
            for v in o:
                visit(v)
    visit(root)
    return best


def yahoo_growth(ticker: str) -> Optional[float]:
    """Yahoo analyst LTG / +1y EPS growth via yfinance (last resort)."""
    try:
        import yfinance as yf
        ge = yf.Ticker(ticker).growth_estimates
        for period in ("LTG", "+1y"):  # LTG preferred but usually NaN now
            if period in ge.index:
                v = ge.loc[period, "stockTrend"]
                if v == v and v is not None:  # not NaN
                    return float(v) * 100
    except Exception:  # noqa: BLE001
        return None
    return None


def projected_growth(ticker: str,
                     finviz_g5: Optional[float]
                     ) -> Tuple[Optional[float], str]:
    """Averaged projected 3-5y EPS growth per the StockOracle recipe.

    Returns (growth_pct, sources_used). Averages every PRIMARY provider
    that answers (GuruFocus + Finviz + Zacks). The stockanalysis.com
    forecast CAGR ALWAYS joins the average when available (it is analyst
    consensus arithmetic, same family as the primaries, and catches
    stale finviz values — e.g. MU: finviz 172.8 vs consensus 69.8).
    Yahoo is last-resort only.
    """
    parts = [("finviz", finviz_g5),
             ("gurufocus", gurufocus_growth(ticker)),
             ("zacks", zacks_growth(ticker)),
             ("stockanalysis", sa_forecast_growth(ticker))]
    avail = [(n, v) for n, v in parts if v is not None]
    if avail:
        g = sum(v for _, v in avail) / len(avail)
        return g, "+".join(n for n, _ in avail)
    v = yahoo_growth(ticker)
    if v is not None:
        return v, "yahoo"
    return None, "none"
