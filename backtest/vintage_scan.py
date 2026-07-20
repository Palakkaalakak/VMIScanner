"""Point-in-time VMI scan for a vintage year.

Uses the REAL scanner pipeline (scanner/vmi: sec.fetch_all + checks.run_checks)
but with fundamental data truncated to fiscal years ending on/before the
vintage date — the scan sees only what an investor standing there could know.
Universe = point-in-time S&P 500 constituents (fja05680 dataset).

No scanner pipeline files are modified; sec.MAX_YEARS is monkeypatched.
Moat is NOT auto-decided (per VMI): scan output feeds a manual wide-moat
pass using Adam's criteria (brand monopoly, switching costs, economies of
scale, barriers to entry, government regulation, high capital costs).

Output: backtest/vintage_scan_<YEAR>.json (checkpointed, resumable).
"""
import concurrent.futures as cf
import json
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scanner"))
from vmi import sec  # noqa: E402
from vmi.checks import run_checks, classify, EXCLUDED_TYPES  # noqa: E402
from vmi.sp500 import fetch_sp500  # noqa: E402
from vmi.http import get  # noqa: E402

sec.MAX_YEARS = 40  # uncap so old fiscal years survive; we truncate ourselves

HERE = os.path.dirname(os.path.abspath(__file__))
VINTAGE_DATES = {2020: "2020-01-02", 2015: "2015-01-02", 2010: "2010-01-04"}


def pit_constituents(vintage_date: str):
    """Tickers in the S&P 500 on the last snapshot on/before vintage_date."""
    import csv
    best = None
    with open(os.path.join(HERE, "sp500_hist.csv")) as f:
        for row in csv.reader(f):
            if row[0] == "date":
                continue
            if row[0] <= vintage_date:
                best = row
            else:
                break
    return best[0], [t.strip() for t in best[1].split(",") if t.strip()]


def truncate(data, cutoff):
    """Keep only fiscal years ending on/before cutoff (point-in-time)."""
    out = {}
    for sect, fd in (data or {}).items():
        if not fd:
            out[sect] = fd
            continue
        fy = fd.get("fiscalYear") or []
        keep = [i for i, f in enumerate(fy) if str(f) <= cutoff]
        out[sect] = {k: [v[i] for i in keep if i < len(v)] for k, v in fd.items()}
    return out


def sic_industry(ticker):
    """SEC submissions sicDescription for tickers absent from current wiki."""
    try:
        cik = sec._ticker_map().get(ticker.upper())
        if cik is None:
            return ""
        raw = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
                  domain_hint="sec", cache_max_age=86400 * 30)
        return json.loads(raw).get("sicDescription") or ""
    except Exception:
        return ""


def scan_vintage(year):
    vd = VINTAGE_DATES[year]
    snap_date, tickers = pit_constituents(vd)
    print(f"[{year}] PIT snapshot {snap_date}: {len(tickers)} tickers", flush=True)

    wiki = {m["ticker"]: m for m in fetch_sp500()}
    out_path = os.path.join(HERE, f"vintage_scan_{year}.json")
    done = {}
    if os.path.exists(out_path):
        done = {r["ticker"]: r for r in json.load(open(out_path))["results"]
                if not r.get("error")}
        print(f"[{year}] resuming: {len(done)} already scanned", flush=True)

    lock = threading.Lock()
    results = dict(done)

    def one(t):
        yft = t.replace(".", "-")
        m = wiki.get(yft) or wiki.get(t)
        if m:
            sector, industry = m["sector"], m.get("sub_industry", "")
        else:
            sector, industry = "", sic_industry(yft)
        ctype = classify(sector, industry)
        base = {"ticker": t, "sector": sector, "industry": industry,
                "company_type": ctype}
        if ctype in EXCLUDED_TYPES:
            return {**base, "excluded": True, "exclusion_reason": ctype}
        try:
            data = sec.fetch_all(yft)
        except Exception as e:
            return {**base, "error": f"fetch: {e}"}
        if not data:
            return {**base, "error": "no SEC data (not a filer / renamed)"}
        dt = truncate(data, vd)
        fy = (dt.get("income") or {}).get("fiscalYear") or []
        if len(fy) < 5:
            return {**base, "error": f"insufficient history: {len(fy)} FYs pre-{year}",
                    "n_years": len(fy)}
        r = run_checks({"ticker": t, "company": t, "sector": sector,
                        "industry": industry}, dt, any_long_window=True)
        d = r.to_dict()
        d.update(base)
        d["n_years"] = len(fy)
        d["fy_span"] = f"{fy[-1][:4]}-{fy[0][:4]}"
        return d

    todo = [t for t in tickers if t not in results]
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(one, t): t for t in todo}
        n = 0
        for fut in cf.as_completed(futs):
            t = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"ticker": t, "error": f"crash: {e}"}
            with lock:
                results[t] = res
                n += 1
                if n % 50 == 0:
                    json.dump({"vintage": year, "date": vd, "snapshot": snap_date,
                               "results": list(results.values())},
                              open(out_path, "w"))
                    print(f"[{year}] {n}/{len(todo)} scanned", flush=True)

    json.dump({"vintage": year, "date": vd, "snapshot": snap_date,
               "results": list(results.values())}, open(out_path, "w"), indent=1)
    greats = [r for r in results.values() if r.get("is_great")]
    errs = sum(1 for r in results.values() if r.get("error"))
    thin = sum(1 for r in results.values()
               if "insufficient history" in str(r.get("error", "")))
    print(f"[{year}] DONE: {len(results)} scanned, {len(greats)} GREAT, "
          f"{errs} errors ({thin} insufficient-history)", flush=True)
    for r in sorted(greats, key=lambda x: -x["score"]):
        print(f"   GREAT {r['ticker']:6} score {r['score']:5} "
              f"({r.get('fy_span')}, {r.get('sector','')[:20]})", flush=True)
    return results


if __name__ == "__main__":
    for y in [int(a) for a in sys.argv[1:]] or [2020]:
        scan_vintage(y)
