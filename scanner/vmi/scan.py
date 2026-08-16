"""Orchestrator: S&P500 universe -> exclude reit/financial/property/commodity
-> per-ticker deep checks (macrotrends.net, 10-15y history) -> JSON output.

Pipeline order ("largest filters first" — see sp500.py docstring for the
long-term multi-universe design this sets up):
  1. Universe: S&P 500 constituents from Wikipedia (sp500.py) — cheap,
     one HTTP call for all 503 tickers + their GICS sector/sub-industry.
  2. FREE exclusion filter: classify() on the GICS sector/sub-industry we
     already have in hand — zero extra network calls. REITs, banks/
     financial firms, property developers and commodity producers are
     dropped here per explicit user instruction, before any expensive
     per-ticker fetch.
  3. (Optional, not used by default) Finviz numeric pre-filter — kept
     available in finviz.py for future large-universe runs where deep-
     checking everything is impractical; skipped for S&P500 since 500
     tickers is small enough to deep-check directly.
  4. Deep fundamental checks via macrotrends.net (up to 15y annual data)
     — the expensive, rate-limited step, run last, only on what survives
     steps 1-3.

Resumability: macrotrends' aggressive rate limiting (8s/request) means a
full S&P500 pass takes hours. Given the sandbox has been wiped mid-run
before, this orchestrator checkpoints progress to the output JSON every
`--checkpoint-every` tickers AND skips tickers already present (with no
error) in an existing output file on startup, so re-running after an
interruption resumes rather than restarting from zero.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from datetime import datetime, timezone

from .checks import run_checks, classify, EXCLUDED_TYPES, ScanResult
from . import macrotrends, sec, yahoo
from .finviz import fetch_growth_estimates
from .sp500 import fetch_sp500
from .dowjones import fetch_dowjones
from .nasdaq100 import fetch_nasdaq100
from .extras import fetch_extras

# repo_root/public/data — served by the Hono dashboard at /data/scan_results.json
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "public", "data")
DEFAULT_OUT = os.path.join(OUT_DIR, "scan_results.json")


# SEC Company Facts first: one fast JSON call per ticker at the SEC's
# documented 10 req/s fair-access rate (every S&P500 name files 10-Ks), so
# a full pass takes ~1-2 min with 8 workers. Yahoo covers OTC non-filers;
# macrotrends (8s/request HTML scrape) is the last-ditch fallback only.
SOURCES = (("sec", sec.fetch_all),
           ("yahoo", yahoo.fetch_all),
           ("macrotrends", macrotrends.fetch_all))


def scan_one(meta: dict, use_cache: bool = True, allow_macrotrends: bool = True) -> ScanResult:
    ticker = meta["ticker"]
    try:
        data, src = None, ""
        for name, fetch in SOURCES:
            if name == "macrotrends" and not allow_macrotrends:
                continue
            try:
                data = fetch(ticker, use_cache=use_cache)
            except Exception:  # noqa: BLE001 — failing source falls through
                data = None
            if data is not None:
                src = name
                break
        if data is None:
            r = ScanResult(ticker=ticker, company=meta.get("company", ""),
                           sector=meta.get("sector", ""), industry=meta.get("industry", ""),
                           company_type=classify(meta.get("sector", ""), meta.get("industry", "")))
            r.error = "no financial data (tried SEC, Yahoo" + (", macrotrends)" if allow_macrotrends else ")")
            return r
        r = run_checks(meta, data,
                       growth_estimate=meta.get("_growth_estimate"),
                       require_5y_only_pass=meta.get("_require_5y_only_pass", False),
                       any_long_window=meta.get("_any_long_window", False))
        r.data_source = src
        return r
    except Exception as e:  # noqa: BLE001
        r = ScanResult(ticker=ticker, company=meta.get("company", ""),
                       sector=meta.get("sector", ""), industry=meta.get("industry", ""))
        r.error = f"{type(e).__name__}: {e}"
        return r


def _load_checkpoint(out_path: str) -> dict:
    """Load an existing output file (if any) to resume from."""
    if not os.path.exists(out_path):
        return {"results": [], "excluded": []}
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return {"results": payload.get("results", []), "excluded": payload.get("excluded", [])}
    except (json.JSONDecodeError, OSError):
        return {"results": [], "excluded": []}


def _write_payload(out_path: str, universe_total: int, included_total: int,
                    excluded_rows: list, results: list, t0: float,
                    universe_label: str = "S&P 500 + Dow Jones 30 "
                                          "(Wikipedia constituents, merged)"):
    great = [r for r in results if not r.get("error") and r.get("is_great")]
    near = [r for r in results if not r.get("error") and not r.get("is_great") and r.get("n_fail", 99) <= 1]
    errors = [r for r in results if r.get("error")]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "criteria_version": "VMI fundamentals only (no valuation/TA) — "
                            "5y window for ROE/margins (explicit in docs), "
                            "10y default elsewhere (docs ambiguous/silent)",
        "universe": universe_label,
        "universe_size": universe_total,
        "included_after_exclusion": included_total,
        "excluded_count": len(excluded_rows),
        "counts": {
            "great": len(great), "near_miss": len(near),
            "failed": len(results) - len(great) - len(near) - len(errors),
            "errors": len(errors),
        },
        "excluded": excluded_rows,
        "results": results,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, out_path)
    return payload


def main():
    ap = argparse.ArgumentParser(description="VMI Great Business Scanner (S&P 500)")
    ap.add_argument("--limit", type=int, default=0, help="limit tickers scanned (0 = all)")
    ap.add_argument("--tickers", type=str, default="",
                     help="comma list to scan directly instead of the S&P500 universe")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--include-dow", dest="include_dow", action="store_true",
                     default=True,
                     help="merge Dow Jones 30 components into the universe (default on)")
    ap.add_argument("--no-dow", dest="include_dow", action="store_false",
                     help="scan the S&P 500 only, without the Dow Jones 30")
    ap.add_argument("--include-ndx", dest="include_ndx", action="store_true",
                     default=True,
                     help="merge Nasdaq-100 constituents into the universe (default on)")
    ap.add_argument("--no-ndx", dest="include_ndx", action="store_false")
    ap.add_argument("--include-extras", dest="include_extras", action="store_true",
                     default=True,
                     help="merge curated non-index quality names (MELI, POOL, "
                          "CNSWF, DSGX, NVO, ASML, ...) into the universe (default on)")
    ap.add_argument("--no-extras", dest="include_extras", action="store_false")
    ap.add_argument("--workers", type=int, default=8,
                     help="parallel fetch workers (SEC fair-access allows 10 req/s)")
    ap.add_argument("--no-macrotrends", dest="allow_macrotrends", action="store_false",
                     default=True, help="skip the slow macrotrends fallback")
    ap.add_argument("--accept-5y-alone", dest="accept_5y_alone", action="store_true",
                     default=False,
                     help="trend/average checks pass if ANY window incl. 5y passes; "
                          "default: a 5y-only pass yields WARN, not PASS")
    ap.add_argument("--any-long-window", dest="any_long_window", action="store_true",
                     default=False,
                     help="trend/average checks pass if ANY of 20/15/10y passes; "
                          "default: only the full 20y window is tested")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--checkpoint-every", type=int, default=25,
                     help="write partial results to --out every N newly-scanned tickers")
    ap.add_argument("--resume", action="store_true", default=True,
                     help="skip tickers already present (without error) in --out (default on)")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    use_cache = not args.no_cache
    out_path = args.out or DEFAULT_OUT

    if args.tickers:
        universe = [{"ticker": t.strip().upper(), "company": "", "sector": "",
                     "industry": "", "country": "", "market_cap": ""}
                    for t in args.tickers.split(",") if t.strip()]
        universe_total = len(universe)
        universe_label = f"ad-hoc tickers ({universe_total})"
    else:
        print("Step 1: fetching S&P 500 constituent list (Wikipedia)...")
        universe = fetch_sp500(use_cache=use_cache)
        sp_n = len(universe)
        print(f"  -> {sp_n} S&P 500 constituents")
        if args.include_dow:
            print("Step 1b: fetching Dow Jones 30 components (Wikipedia)...")
            try:
                dow = fetch_dowjones(use_cache=use_cache)
                seen = {m["ticker"] for m in universe}
                added = [m for m in dow if m["ticker"] not in seen]
                universe.extend(added)
                print(f"  -> {len(dow)} DJIA components, "
                      f"{len(added)} not already in S&P500 (merged)")
            except Exception as e:
                print(f"  !! DJIA fetch failed ({e}) — continuing with S&P500 only")
        if args.include_ndx:
            print("Step 1c: fetching Nasdaq-100 constituents (stockanalysis.com)...")
            try:
                ndx = fetch_nasdaq100(use_cache=use_cache)
                seen = {m["ticker"] for m in universe}
                added = [m for m in ndx if m["ticker"] not in seen]
                universe.extend(added)
                print(f"  -> {len(ndx)} NDX constituents, "
                      f"{len(added)} not already in universe (merged)")
            except Exception as e:
                print(f"  !! NDX fetch failed ({e}) — continuing without it")
        if args.include_extras:
            extras = fetch_extras()
            seen = {m["ticker"] for m in universe}
            added = [m for m in extras if m["ticker"] not in seen]
            universe.extend(added)
            print(f"Step 1d: curated extras (non-index quality: MELI, POOL, "
                  f"CNSWF, DSGX, NVO, ASML, ...): {len(extras)} names, "
                  f"{len(added)} new (merged)")
        universe_total = len(universe)
        parts = ["S&P 500"]
        if args.include_dow:
            parts.append("Dow Jones 30")
        if args.include_ndx:
            parts.append("Nasdaq-100")
        if args.include_extras:
            parts.append("curated extras (ADRs/ex-index quality)")
        universe_label = " + ".join(parts) + " (merged)"
        print(f"  -> universe: {universe_total} tickers ({universe_label})")

    # ---- Step 2: FREE exclusion filter (no network calls) ----
    included = []
    excluded_rows = []
    for m in universe:
        ctype = classify(m.get("sector", ""), m.get("industry", ""))
        if ctype in EXCLUDED_TYPES:
            excluded_rows.append({
                "ticker": m["ticker"], "company": m.get("company", ""),
                "sector": m.get("sector", ""), "industry": m.get("industry", ""),
                "company_type": ctype,
                "reason": f"Excluded: {ctype} — ETFs are funds, not companies; "
                          "no financial statements to check. (Banks/REITs are "
                          "INCLUDED since 2026-08-16 with their own VMI "
                          "methods: P/B §8, P/NAV+yield §9, DNI §7.)",
            })
        else:
            included.append(m)
    print(f"Step 2: excluded {len(excluded_rows)} ETFs "
          f"-> {len(included)} remain to deep-scan (banks/REITs now included)")

    if args.limit:
        included = included[:args.limit]

    # ---- Resume support ----
    prior = _load_checkpoint(out_path) if args.resume else {"results": [], "excluded": []}
    prior_by_ticker = {r["ticker"]: r for r in prior["results"] if not r.get("error")}
    todo = [m for m in included if m["ticker"] not in prior_by_ticker]
    already_done = len(included) - len(todo)
    if already_done:
        print(f"Resuming: {already_done} tickers already scanned successfully in "
              f"{out_path}, {len(todo)} remaining")

    results = list(prior_by_ticker.values())
    # Preserve excluded_rows across resumes without duplicating — but DROP
    # stale entries whose company_type is no longer excluded (e.g. the 60
    # bank/REIT rows carried over from checkpoints made before the
    # 2026-08-16 valuation-routing change) or that were actually scanned.
    if prior.get("excluded"):
        seen_excl = {r["ticker"] for r in excluded_rows}
        for r in prior["excluded"]:
            if (r["ticker"] not in seen_excl
                    and r.get("company_type") in EXCLUDED_TYPES
                    and r["ticker"] not in prior_by_ticker):
                excluded_rows.append(r)
                seen_excl.add(r["ticker"])

    # Forward growth estimates: finviz bulk pull (~25 pages for the whole
    # S&P500, not per-ticker) so check 13 stops being NA.
    growth_by_ticker = {}
    try:
        print("Step 2b: fetching analyst EPS estimates (finviz bulk)...")
        growth_by_ticker = fetch_growth_estimates(
            [m["ticker"] for m in todo], use_cache=use_cache)
        print(f"  -> estimates for "
              f"{sum(1 for v in growth_by_ticker.values() if v)}/{len(todo)} tickers")
    except Exception as e:  # noqa: BLE001 — enrichment, not fatal
        print(f"  finviz estimates unavailable ({type(e).__name__}: {e})")
    for m in todo:
        m["_growth_estimate"] = growth_by_ticker.get(m["ticker"])
        m["_require_5y_only_pass"] = not args.accept_5y_alone
        m["_any_long_window"] = args.any_long_window

    print(f"Step 3: deep fundamental checks on {len(todo)} tickers "
          f"(SEC-first, {args.workers} workers; Yahoo/macrotrends fallbacks)...")
    t0 = time.time()
    done_new = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(scan_one, m, use_cache, args.allow_macrotrends): m["ticker"]
                for m in todo}
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r.to_dict())
            done_new += 1
            if done_new % 25 == 0 or done_new == len(todo):
                great = sum(1 for x in results if not x.get("error") and x.get("is_great"))
                print(f"  {done_new}/{len(todo)} new tickers scanned "
                      f"({already_done + done_new}/{len(included)} total, "
                      f"{great} great so far, {time.time()-t0:.0f}s elapsed this run)")
            if done_new % args.checkpoint_every == 0:
                _write_payload(out_path, universe_total, len(included), excluded_rows, results, t0,
                               universe_label)
                print(f"  [checkpoint written to {out_path}]")

    results.sort(key=lambda r: (-r.get("score", 0), r.get("n_fail", 99), r["ticker"]))
    payload = _write_payload(out_path, universe_total, len(included), excluded_rows, results, t0,
                             universe_label)

    print(f"\n=== DONE (this run: {time.time()-t0:.0f}s) ===")
    print(f"Great businesses (0 fails): {payload['counts']['great']}")
    print(f"Near misses (1 fail):       {payload['counts']['near_miss']}")
    print(f"Errors:                     {payload['counts']['errors']}")
    print(f"Excluded (ETFs only — banks/REITs are scanned): {payload['excluded_count']}")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
