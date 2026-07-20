# VMI 2000–2013 Backtest (separate from the scanner — does not touch scanner code)

Two $1M accounts started Jan 2000, run to Dec 2013, weekly adjusted closes
(dividends reinvested). Zero dot-com/tech/telecom/internet exposure.

## Method
- **Selection (no hindsight beyond sector tilt)**: VMI great-business checklist
  on 1990s fundamentals; wide-moat only; DCF-undervalued only.
- **DCF**: course-style 20y — years 1–10 at g, years 11–20 at 4%, no terminal
  value; discount = rf 6.5% (Jan-2000 10y Treasury) + beta × 4% MRP; base flow =
  trailing EPS × OCF-conversion. IV refreshed annually at min(g, 10%)/yr.
- **Entries**: tranche 1 first week price < IV; adds only at/below the 40-week
  SMA AND under IV, ≥8 weeks apart, max 3 tranches = 6.25% cap (16 stocks).
- **Sells**: only fraud/scandal — BMY Jul-2002 (accounting) → PFE; UNH Oct-2006
  (options backdating) → KO; CAH Jul-2004 (SEC probe) → GPC. MRK held through
  Vioxx (product withdrawal ≠ fraud; checklist intact).

## Accounts
- **A DEFENSIVE**: JNJ ABT BMY BDX UNH MRK · PG KMB CLX GIS HSY HRL PEP CL · MCD GD
- **B GROWTH**: TJX ROST AZO ORLY NKE SYY CAH ITW DHR LOW CVS TGT DLTR CHD SBUX SYK

## Results (2000-01-07 → 2013-12-31)
| | Total return | CAGR | Final value | 7%+ corrections |
|---|---|---|---|---|
| **Growth (B)** | **+813.8%** | **17.16%** | $9,137,716 | 12 (max −34.4%) |
| **Defensive (A)** | **+330.6%** | **11.02%** | $4,305,880 | 4 (max −32.7%) |
| S&P 500 (div. reinv.) | +63.1% | 3.56% | $1,631,089 | 6 (max −54.4%) |

## Files
- `simulate2.py` — the simulation (era P/E, growth, beta, OCF-mult documented inline)
- `plot2.py` → `vmi_2000_2013_two_accounts.png` — equity curves + drawdown panel
- `stats2.json` — CAGR/total/final + every 7%+ correction with decline/recovery days
- `trades_defensive.txt`, `trades_growth.txt` — full trade logs
- `eq_*.csv` — weekly equity curves; `weekly_adj.csv` — price data (Yahoo, adjusted)
