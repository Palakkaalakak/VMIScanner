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
from . import macrotrends
from .sp500 import fetch_sp500

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public", "data")
DEFAULT_OUT = os.path.join(OUT_DIR, "scan_results.json")


def scan_one(meta: dict, use_cache: bool = True) -> ScanResult:
    ticker = meta["ticker"]
    try:
        data = macrotrends.fetch_all(ticker, use_cache=use_cache)
        if data is None:
            r = ScanResult(ticker=ticker, company=meta.get("company", ""),
                           sector=meta.get("sector", ""), industry=meta.get("industry", ""),
                           company_type=classify(meta.get("sector", ""), meta.get("industry", "")))
            r.error = "no financial data on macrotrends.net"
            return r
        return run_checks(meta, data, has_growth_prefilter=meta.get("_has_growth_prefilter", False))
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
                    excluded_rows: list, results: list, t0: float):
    great = [r for r in results if not r.get("error") and r.get("is_great")]
    near = [r for r in results if not r.get("error") and not r.get("is_great") and r.get("n_fail", 99) <= 1]
    errors = [r for r in results if r.get("error")]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "criteria_version": "VMI fundamentals only (no valuation/TA) — "
                            "5y window for ROE/margins (explicit in docs), "
                            "10y default elsewhere (docs ambiguous/silent)",
        "universe": "S&P 500 (Wikipedia constituents)",
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
    ap.add_argument("--workers", type=int, default=1,
                     help="macrotrends is globally throttled to 1 req/8s regardless of "
                          "worker count, so >1 worker will not speed up a macrotrends-"
                          "only run — kept configurable for future faster sources")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--checkpoint-every", type=int, default=15,
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
    else:
        print("Step 1: fetching S&P 500 constituent list (Wikipedia)...")
        universe = fetch_sp500(use_cache=use_cache)
        universe_total = len(universe)
        print(f"  -> {universe_total} S&P 500 constituents")

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
                "reason": f"Excluded per user instruction: {ctype} companies have "
                          "VMI exception rules (REITs/banks/financial firms/property/"
                          "commodity are structurally leveraged) — not scanned for now.",
            })
        else:
            included.append(m)
    print(f"Step 2: excluded {len(excluded_rows)} reit/financial/property/commodity "
          f"companies -> {len(included)} remain to deep-scan")

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
    # Preserve excluded_rows across resumes without duplicating.
    if prior.get("excluded"):
        seen_excl = {r["ticker"] for r in excluded_rows}
        for r in prior["excluded"]:
            if r["ticker"] not in seen_excl:
                excluded_rows.append(r)
                seen_excl.add(r["ticker"])

    print(f"Step 3: deep fundamental checks (macrotrends.net) on {len(todo)} tickers "
          f"(~8s/request x 4 statements/ticker — this is slow by design, macrotrends "
          "rate-limits aggressively)...")
    t0 = time.time()
    done_new = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(scan_one, m, use_cache): m["ticker"] for m in todo}
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r.to_dict())
            done_new += 1
            if done_new % 5 == 0 or done_new == len(todo):
                great = sum(1 for x in results if not x.get("error") and x.get("is_great"))
                print(f"  {done_new}/{len(todo)} new tickers scanned "
                      f"({already_done + done_new}/{len(included)} total, "
                      f"{great} great so far, {time.time()-t0:.0f}s elapsed this run)")
            if done_new % args.checkpoint_every == 0:
                _write_payload(out_path, universe_total, len(included), excluded_rows, results, t0)
                print(f"  [checkpoint written to {out_path}]")

    results.sort(key=lambda r: (-r.get("score", 0), r.get("n_fail", 99), r["ticker"]))
    payload = _write_payload(out_path, universe_total, len(included), excluded_rows, results, t0)

    print(f"\n=== DONE (this run: {time.time()-t0:.0f}s) ===")
    print(f"Great businesses (0 fails): {payload['counts']['great']}")
    print(f"Near misses (1 fail):       {payload['counts']['near_miss']}")
    print(f"Errors:                     {payload['counts']['errors']}")
    print(f"Excluded (reit/fin/prop/commodity): {payload['excluded_count']}")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
