"""Orchestrator: finviz pre-filter -> per-ticker deep checks -> JSON output."""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from datetime import datetime, timezone

from .checks import run_checks, ScanResult
from .finviz import screen_universe
from .stockanalysis import fetch_all, fetch_profile

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public", "data")


def scan_one(meta: dict, use_cache: bool = True) -> ScanResult:
    ticker = meta["ticker"]
    try:
        if not meta.get("sector") or not meta.get("industry"):
            prof = fetch_profile(ticker, use_cache=use_cache)
            meta = {**meta,
                    "sector": meta.get("sector") or prof.get("sector", ""),
                    "industry": meta.get("industry") or prof.get("industry", "")}
        data = fetch_all(ticker, use_cache=use_cache)
        if data is None:
            r = ScanResult(ticker=ticker, company=meta.get("company", ""))
            r.error = "no financial data on stockanalysis.com"
            return r
        return run_checks(meta, data)
    except Exception as e:  # noqa: BLE001
        r = ScanResult(ticker=ticker, company=meta.get("company", ""),
                       sector=meta.get("sector", ""), industry=meta.get("industry", ""))
        r.error = f"{type(e).__name__}: {e}"
        return r


def main():
    ap = argparse.ArgumentParser(description="VMI Great Business Scanner")
    ap.add_argument("--limit", type=int, default=0, help="limit tickers (0 = all)")
    ap.add_argument("--tickers", type=str, default="", help="comma list to scan directly")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    use_cache = not args.no_cache

    if args.tickers:
        universe = [{"ticker": t.strip().upper()} for t in args.tickers.split(",") if t.strip()]
    else:
        print("Step 1: Finviz pre-filter (VMI screen minus valuation filters)...")
        universe = screen_universe(use_cache=use_cache)
        print(f"  -> {len(universe)} candidates pass the pre-filter")

    if args.limit:
        universe = universe[:args.limit]

    print(f"Step 2: deep fundamental checks on {len(universe)} tickers...")
    results = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(scan_one, m, use_cache): m["ticker"] for m in universe}
        done = 0
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            if done % 25 == 0 or done == len(universe):
                great = sum(1 for x in results if not x.error and x.is_great)
                print(f"  {done}/{len(universe)} scanned "
                      f"({great} great so far, {time.time()-t0:.0f}s)")

    results.sort(key=lambda r: (-r.score, r.n_fail, r.ticker))

    great = [r for r in results if not r.error and r.is_great]
    near = [r for r in results if not r.error and not r.is_great and r.n_fail <= 1]
    errors = [r for r in results if r.error]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "criteria_version": "VMI Lessons 4+7 fundamentals (no valuation/TA)",
        "universe_size": len(universe),
        "counts": {"great": len(great), "near_miss": len(near),
                   "failed": len(results) - len(great) - len(near) - len(errors),
                   "errors": len(errors)},
        "results": [r.to_dict() for r in results],
    }

    out_path = args.out or os.path.join(OUT_DIR, "scan_results.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)

    print(f"\n=== DONE in {time.time()-t0:.0f}s ===")
    print(f"Great businesses (0 fails): {len(great)}")
    print(f"Near misses (1 fail):       {len(near)}")
    print(f"Errors:                     {len(errors)}")
    print(f"Output: {out_path}")
    if great:
        print("\nTop great businesses by score:")
        for r in great[:40]:
            print(f"  {r.ticker:<7} {r.score:>5}%  {r.company[:38]:<38} {r.sector}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
