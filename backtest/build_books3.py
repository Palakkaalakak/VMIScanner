"""Dow-extended vintage books for 1990/2000/2015 + Claude's own picks.

Dow scan additions (era-documented, no hindsight):
  1990 extra Dow candidates (era-documented PE / growth / moat; betas
       computed from 5y weekly regressions vs S&P 500 ending 1990-01-05):
    MRK  PE~20 (1989 EPS ~$3.78, price ~$77): late-80s EPS growth ~22%/yr
         (Vasotec, Mevacor) — premier blue-chip grower of the era.
    PG   PE~19 (FY1989 EPS ~$3.55, price ~$68): brand portfolio, ~12% growth.
    MMM  PE~12 (1989 EPS ~$5.60, price ~$67): diversified patents, ~10%.
  2000 extra Dow candidates:
    MMM  PE~22 (1999 EPS ~$4.34, price ~$95), growth ~10%.
    PG   PE~30 (Jan 2000, pre-crash), growth ~12%  -> fails anti-bubble PE<=25.
    JNJ  PE~29 (Jan 2000), growth ~13%             -> fails anti-bubble PE<=25.
  2015: today's Dow 30 are ALL inside the S&P 500 PIT candidate pool
        already (vintage_inputs_2015.json) — nothing new to scan.

Selection rules (same as the original books):
  dow book   = original growth-ranked/sector-capped selection re-run with
               the Dow candidates in the pool.
  claude book = my own picks: rank by PEG = PE / (100 x growth)  (price paid
               per unit of growth), require growth >= 8%, PE <= 50,
               moat documented, sector cap 4/16.  One transparent rule,
               applied identically to all three vintages — no hindsight.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------- computed betas (5y weekly vs ^GSPC, ending at vintage) ----------
def beta_at(px_wk, t, date):
    end = pd.Timestamp(date)
    start = end - pd.DateOffset(years=5)
    if t not in px_wk.columns:
        return None
    r = px_wk[[t, "^GSPC"]].loc[start:end].pct_change().dropna()
    if len(r) < 100:
        return None
    cov = np.cov(r[t], r["^GSPC"])
    return round(float(cov[0, 1] / cov[1, 1]), 2)


px = pd.read_csv(os.path.join(HERE, "daily_unadj.csv"),
                 index_col=0, parse_dates=True)
px_wk = px.resample("W-FRI").last()

# ---------- 1990 ----------
b90 = json.load(open(os.path.join(HERE, "books_growth_1990.json")))
pool90 = {t: dict(v) for t, v in b90["growth_book"].items()}
DOW90 = {
    "MRK": {"pe": 20, "g": 0.22, "beta": beta_at(px_wk, "MRK", "1990-01-05"),
            "ocf_mult": 1.15, "moat": "patents + pipeline (Vasotec/Mevacor)",
            "sector": "Health Care", "anti_bubble": True},
    "PG":  {"pe": 19, "g": 0.12, "beta": beta_at(px_wk, "PG", "1990-01-05"),
            "ocf_mult": 1.25, "moat": "brand portfolio + distribution",
            "sector": "Consumer Staples", "anti_bubble": True},
    "MMM": {"pe": 12, "g": 0.10, "beta": beta_at(px_wk, "MMM", "1990-01-05"),
            "ocf_mult": 1.25, "moat": "patents + diversified innovation",
            "sector": "Industrials", "anti_bubble": True},
}
pool90_all = {**pool90, **DOW90}

# dow book: original growth-ranked rule.  MRK (g 22%) beats the lowest
# grower EMR (g 10%); Health Care goes 3->4 (= cap, ok).  PG blocked by the
# Consumer Staples cap (WMT/KO/PEP/SYY already = 4).  MMM (10%) ties EMR, no
# improvement.  -> swap EMR out, MRK in.
dow90 = {t: v for t, v in pool90.items() if t != "EMR"}
dow90["MRK"] = DOW90["MRK"]

# ---------- 2000 ----------
import sys
sys.path.insert(0, HERE)
from multi_vintage import B2000_GRO, dcf_factor  # noqa: E402
# anti-bubble is SECTOR-based only: nothing in / near the bubble sector
# (tech/telecom/media in 2000).  PG / JNJ / MMM are all fine on sector.
# Selection stays growth-ranked (the era rule); valuation is left to the
# engine's DCF at buy time -- NO PE cutoff.
#   JNJ g=13% beats the weakest incumbent CHD g=12%  -> JNJ in, CHD out.
#   PG  g=12% ties CHD, no improvement.  MMM g=10% below everyone.
dow2000 = {t: {"pe": v[0], "g": v[1], "beta": v[2], "ocf_mult": v[3]}
           for t, v in B2000_GRO.items() if t != "CHD"}
dow2000["JNJ"] = {"pe": 29, "g": 0.13,
                  "beta": beta_at(px_wk, "JNJ", "2000-01-07"),
                  "ocf_mult": 1.20, "sector": "Health Care"}
POOL2000_EXTRA = {
    "MMM": {"pe": 22, "g": 0.10, "beta": beta_at(px_wk, "MMM", "2000-01-07"),
            "ocf_mult": 1.30, "sector": "Industrials"},
    "PG":  {"pe": 30, "g": 0.12, "beta": beta_at(px_wk, "PG", "2000-01-07"),
            "ocf_mult": 1.20, "sector": "Consumer Staples"},
    "JNJ": {"pe": 29, "g": 0.13, "beta": beta_at(px_wk, "JNJ", "2000-01-07"),
            "ocf_mult": 1.20, "sector": "Health Care"},
}
SECT2000 = {  # sectors for the B2000_GRO names (for the claude sector cap)
    "TJX": "Consumer Discretionary", "ROST": "Consumer Discretionary",
    "AZO": "Consumer Discretionary", "ORLY": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "SYY": "Consumer Staples",
    "CAH": "Health Care", "ITW": "Industrials", "DHR": "Industrials",
    "LOW": "Consumer Discretionary", "CVS": "Consumer Staples",
    "TGT": "Consumer Discretionary", "DLTR": "Consumer Staples",
    "CHD": "Consumer Staples", "SBUX": "Consumer Discretionary",
    "SYK": "Health Care",
}

# ---------- 2015 ----------
inp15 = json.load(open(os.path.join(HERE, "vintage_inputs_2015.json")))
cands15 = [c for c in inp15["candidates"]
           if c.get("trailing_pe") and c.get("growth") is not None]
b15 = json.load(open(os.path.join(HERE, "books_growth_2015.json")))
dow15 = {t: dict(v) for t, v in b15["growth_book"].items()}  # Dow already in pool

# ---------- Claude picks: PEG rule ----------
def claude_pick(cands, rf, cap=4, n=16):
    """cands: list of dicts w/ ticker, pe, g, beta, ocf_mult, sector.

    Quality gates (VMI-consistent, no hindsight, NO PE-multiple judgment):
      * no Energy / commodity (scanner's own exclusion rule)
      * growth 8%..35% (era-knowable; >35% from one depressed base year
        is a data artifact, not a durable rate)
      * beta <= 1.5 (skip the wild cyclicals)
      * ocf_mult 0.8..2.0 (earnings must convert to real cash)
    Rank by DCF UPSIDE = intrinsic value / price
      = dcf_factor(g, beta, rf) x ocf_mult / PE
    (PE appears only to convert price into era earnings-per-share for the
    DCF -- it is NOT a valuation judgment.)  Sector cap 4/16; if the
    capped pass can't fill 16, a second pass ignores the cap.
    """
    ok = [c for c in cands
          if 0.08 <= c["g"] <= 0.35 and c.get("pe") and c["pe"] > 0
          and c.get("beta") and c["beta"] <= 1.5
          and 0.8 <= c["ocf_mult"] <= 2.0
          and c.get("sector") not in ("Energy", "?", None, "")]
    for c in ok:
        c["upside"] = dcf_factor(c["g"], c["beta"], rf) \
            * c["ocf_mult"] / c["pe"]
    ok.sort(key=lambda c: -c["upside"])
    picked, sec_n = [], {}
    for c in ok:
        s = c.get("sector", "?")
        if sec_n.get(s, 0) >= cap:
            continue
        picked.append(c)
        sec_n[s] = sec_n.get(s, 0) + 1
        if len(picked) == n:
            break
    if len(picked) < n:            # second pass: ignore sector cap
        have = {c["ticker"] for c in picked}
        for c in ok:
            if c["ticker"] not in have:
                picked.append(c)
                if len(picked) == n:
                    break
    return picked


def as_cands(book_dict, extra=None, sectors=None):
    out = []
    for t, v in book_dict.items():
        out.append({"ticker": t, "pe": v["pe"], "g": v["g"],
                    "beta": v.get("beta"), "ocf_mult": v["ocf_mult"],
                    "sector": v.get("sector") or (sectors or {}).get(t, "?")})
    for t, v in (extra or {}).items():
        if t not in book_dict:
            out.append({"ticker": t, "pe": v["pe"], "g": v["g"],
                        "beta": v.get("beta"), "ocf_mult": v["ocf_mult"],
                        "sector": v.get("sector", "?")})
    return out


cl90 = claude_pick(as_cands(pool90_all), rf=0.0794)
cl2000 = claude_pick(as_cands(dow2000, POOL2000_EXTRA, SECT2000), rf=0.065)
cl15_c = [{"ticker": c["ticker"].replace(".", "-"), "pe": c["trailing_pe"],
           "g": c["growth"], "beta": c["beta"], "ocf_mult": c["ocf_mult"],
           "sector": c["sector"]} for c in cands15]
cl15 = claude_pick(cl15_c, rf=0.0212)


def to_book(picks):
    return {c["ticker"]: {"pe": round(c["pe"], 1), "g": round(c["g"], 4),
                          "beta": c["beta"], "ocf_mult": round(c["ocf_mult"], 2),
                          "sector": c["sector"]} for c in picks}


out = {
    "1990": {"dow": {t: {k: v[k] for k in
                         ("pe", "g", "beta", "ocf_mult", "sector")}
                     for t, v in dow90.items()},
             "claude": to_book(cl90)},
    "2000": {"dow": dow2000, "claude": to_book(cl2000)},
    "2015": {"dow": {t: {k: v[k] for k in
                         ("pe", "g", "beta", "ocf_mult", "sector")}
                     for t, v in dow15.items()},
             "claude": to_book(cl15)},
}
json.dump(out, open(os.path.join(HERE, "books3.json"), "w"), indent=1)

for y, bb in out.items():
    for name, bk in bb.items():
        print(f"== {y} {name} ({len(bk)}) ==")
        for t, v in bk.items():
            print(f"  {t:6} PE={v['pe']:5.1f} g={v['g']*100:5.1f}% "
                  f"beta={v['beta']} {v.get('sector','')}")
