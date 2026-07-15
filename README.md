# VMI Great Business Scanner

## Project Overview
- **Name**: VMI Great Business Scanner
- **Goal**: Automatically screen the US stock market for "great businesses" as
  defined by Adam Khoo's Value Momentum Investing (VMI) course (Piranha
  Profits / Wealth Academy), using only **free** data sources. The scanner
  deliberately covers **only Step 1 of the VMI framework** ("Is it a great
  business?") — valuation (intrinsic value) and technical analysis (entry
  timing) are out of scope by design; those require human judgment on price
  and are meant to happen *after* this quality filter.
- **Moat is never auto-scored.** The course treats "sustainable competitive
  advantage" as a qualitative judgment call, so the scanner surfaces
  *moat hints* (margin levels, ROIC persistence, buyback yield) instead of
  a fabricated score, and leaves the final moat verdict to the user.

## How it works (pipeline)
1. **Universe pre-filter — Finviz screener** (`scanner/vmi/finviz.py`)
   Scrapes `finviz.com/screener.ashx` with the VMI-recommended filter set
   (Lesson 9 / Quick Reference J2), minus the valuation filter (PEG), since
   this tool is fundamentals-only:
   - Sales growth past 5 years: positive
   - EPS growth past 5 years / this year / next year / next 5 years: positive
   - ROE > 10%
   - Current ratio > 1
2. **Deep fundamental checks — stockanalysis.com** (`scanner/vmi/stockanalysis.py`, `checks.py`)
   For every ticker that survives the pre-filter, fetches 10 years of
   Income Statement, Balance Sheet, Cash Flow Statement and Ratios from
   stockanalysis.com's public SvelteKit data endpoints, then evaluates the
   full VMI checklist (Lessons 4 & 7):
   - Sales / Net Income (or Operating Income fallback) / CFO consistently increasing (5y)
   - Free Cash Flow consistently positive
   - Gross & Net margin consistent or increasing
   - ROE ≥ 12%, ROIC ≥ 12% (n/a banks)
   - Current ratio ≥ 1, Debt/EBITDA ≤ 3, Debt servicing ratio < 30%
   - REITs: Gearing (Total Debt/Total Assets) < 45% instead of the standard debt ratios
   - Banks/financials: CET1 & NPL flagged "check manually" (no free source found)
   - Receivables growing no faster than sales revenue (channel-stuffing red flag)
   - Positive projected growth (guaranteed by the Finviz pre-filter)
   A company qualifies as **"Great Business"** when it has **zero FAILs**
   among the checks applicable to its business type (banks/REITs/property/
   commodity companies get contextual exemptions exactly as the course
   specifies).
3. **Dashboard** (`src/index.tsx`, `public/static/app.js`) — a Hono +
   vanilla-JS single page that reads `public/data/scan_results.json` and
   lets you search/filter/sort and drill into each ticker's full checklist
   and moat hints.

## Running the scanner
```bash
cd scanner
pip install -r requirements.txt

# Full scan (all Finviz pre-filter candidates, ~400+ tickers)
python3 -m vmi.scan --workers 6 --out ../public/data/scan_results.json

# Quick test on specific tickers
python3 -m vmi.scan --tickers "AAPL,MSFT,GOOGL,JPM,O" --out /tmp/test.json

# Force-refresh cached HTTP responses
python3 -m vmi.scan --no-cache
```
Results are cached on disk (`scanner/cache/`) for 1-3 days to be polite to
the free sources and speed up re-runs; delete the cache or pass `--no-cache`
to force fresh data.

## Data sources (all free, no API keys required)
| Purpose | Source | Method |
|---|---|---|
| Universe pre-filter | finviz.com/screener.ashx | HTML scraping |
| 10y financial statements & ratios | stockanalysis.com | SvelteKit `__data.json` scraping |
| Sector/Industry classification | stockanalysis.com company profile | SvelteKit `__data.json` scraping |

## URLs
- **GitHub**: https://github.com/Palakkaalakak/VMIScanner
- **Local dev**: http://localhost:3000 (via PM2 + wrangler pages dev)

## Data Architecture
- **Storage**: static JSON file (`public/data/scan_results.json`), regenerated
  by re-running the Python scanner — no database needed since this is a
  point-in-time screen, not a live-tracked portfolio.
- **Data model**: see `ScanResult`/`CheckResult` dataclasses in `scanner/vmi/checks.py`.

## User Guide
1. Open the dashboard.
2. Use the **Result** filter to switch between "Great Businesses" (0 fails),
   "Near Misses" (1 fail — worth a second look), or "All Scanned".
3. Filter by sector or company type (standard / financial / REIT / property / commodity).
4. Click any row to see the full 12-13 point checklist with PASS/FAIL/WARN/NA
   status, plus **moat hints** you should evaluate yourself before investing.
5. This tool answers "is it a great business?" only. You still need to do
   your own valuation (intrinsic value vs. price) and technical analysis
   (entry timing) before making any investment decision — per the VMI 3-step framework.

## Features not yet implemented
- Non-US exchanges (HK/SG/etc.) — Finviz only covers US-listed tickers;
  aastocks.com scraping for HK stocks would need a separate pre-filter.
- Bank CET1/NPL ratios — no free source found; flagged for manual check.
- Historical re-runs / trend tracking across scan dates.
- Scheduled re-scans (currently a manual CLI run).

## Recommended next steps
- Add a scheduled Cloudflare Cron Trigger + Worker that shells out is not
  possible on Workers (no Python runtime) — the scan must keep running from
  a machine with Python (this sandbox, or any CI runner), with the resulting
  JSON re-deployed to Cloudflare Pages.
- Consider an HK/SG universe pass via aastocks.com for the course's
  Singapore/HK examples (UOB, Tencent, JD.com, CapitaLand REIT, etc.).

## Deployment
- **Platform**: Cloudflare Pages (Hono framework)
- **Status**: Dashboard built and tested locally; not yet deployed to Cloudflare
- **Tech Stack**: Hono + TypeScript (frontend shell) + Python 3 scanner (data pipeline) + Tailwind CSS (CDN)
- **Last Updated**: 2026-07-15
