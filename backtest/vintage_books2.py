"""Growth-only, sector-capped book rebuild for 2015 / 2020 vintages.

User rules (this pass):
  * accounts are ALL GROWTH — no defensive book
  * <= 25% of the book in any single GICS sector (16 stocks -> max 4/sector)
    (exception allowed only when deliberately preparing for a known
     sector-specific bubble, e.g. the 2000 dot-com book — handled separately)
  * anti-bubble slice unchanged: >= 8/16 at era trailing PE <= 25,
    nothing above PE 50 ever enters a book
  * picks come from the point-in-time scanner GREAT list, filtered by the
    manual era wide-moat pass (WIDE_2015 / WIDE_2020 in vintage_books.py),
    ranked by highest era growth (5y sales CAGR first).
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vintage_books import WIDE_2015, WIDE_2020, PE_VALUE, PE_BUBBLE  # noqa: E402

SECTOR_CAP = 4  # 25% of 16


def build(year):
    wide = {2015: WIDE_2015, 2020: WIDE_2020}[year]
    d = json.load(open(os.path.join(HERE, f"vintage_inputs_{year}.json")))
    cands = {c["ticker"]: c for c in d["candidates"]
             if c.get("trailing_pe") and c.get("growth") is not None
             and c.get("beta") is not None and c.get("ocf_mult")}
    pool = [(t, cands[t]) for t in wide if t in cands
            and cands[t]["trailing_pe"] <= PE_BUBBLE]
    pool.sort(key=lambda x: -x[1]["growth"])

    picked, secs = [], Counter()

    def sector_of(c):
        return c.get("sector") or "Other"

    # phase 1: 8 anti-bubble slots (PE <= 25), highest growth, sector-capped
    for t, c in pool:
        if len(picked) >= 8:
            break
        if c["trailing_pe"] <= PE_VALUE and secs[sector_of(c)] < SECTOR_CAP:
            picked.append((t, c)); secs[sector_of(c)] += 1
    # phase 2: fill to 16 from the rest (PE <= 50), highest growth, capped
    for t, c in pool:
        if len(picked) >= 16:
            break
        if any(t == p for p, _ in picked):
            continue
        if secs[sector_of(c)] < SECTOR_CAP:
            picked.append((t, c)); secs[sector_of(c)] += 1

    assert len(picked) == 16, f"{year}: only {len(picked)} picks"
    ab = sum(1 for _, c in picked if c["trailing_pe"] <= PE_VALUE)
    assert ab >= 8, f"{year}: anti-bubble {ab}/16 < 8"
    assert max(secs.values()) <= SECTOR_CAP

    book = {t: {"pe": round(c["trailing_pe"], 1), "g": round(c["growth"], 4),
                "beta": c["beta"], "ocf_mult": c["ocf_mult"],
                "moat": wide[t], "sector": sector_of(c),
                "anti_bubble": c["trailing_pe"] <= PE_VALUE}
            for t, c in picked}
    out = {"vintage": year, "rf": d["rf_10y"], "mrp": 0.04,
           "sector_cap": SECTOR_CAP, "growth_book": book,
           "anti_bubble_count": ab,
           "sector_mix": dict(secs)}
    json.dump(out, open(os.path.join(HERE, f"books_growth_{year}.json"), "w"),
              indent=1)
    print(f"== {year} growth (sector-capped) ==  anti-bubble {ab}/16")
    for t, c in picked:
        print(f"  {t:6} g={c['growth']*100:5.1f}%  PE={c['trailing_pe']:5.1f} "
              f"beta={c['beta']:4.2f}  {sector_of(c)[:22]:22} "
              f"{'*' if c['trailing_pe'] <= PE_VALUE else ' '} {wide[t][:40]}")
    print("  sectors:", dict(secs))


if __name__ == "__main__":
    build(2015)
    build(2020)
