#!/usr/bin/env python3
"""Rebuild backtest/buylist_2026-07-31.json - REV 3 "roi des rois".

Only category KINGS allowed (best business of its type, best of the best).
IV source hierarchy: Adam's watchlist IVs (user-supplied screenshots,
Base IV column) override scanner DCF; divergences documented.
2000-account rules: 6.25% cap, 3 tranches, adds at 200d SMA x1.01 under IV
>= 56d apart, never trim. PMCC if g<=15%, plain shares if g>15%.
"""
import json, os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN = os.path.join(BASE, "public", "data", "scan_results.json")
OUT = os.path.join(BASE, "backtest", "buylist_2026-07-31.json")

# Base IVs transcribed from Adam's watchlist screenshots (2026-07-31 upload)
ADAM_IV = {
    "PEP": 158.00, "MCD": 273.06, "DPZ": 421.11, "FDS": 334.00,
    "CME": 221.00, "MCO": 467.00, "MNST": 62.03, "ISRG": 269.94,
    "COST": 576.00, "BRK-B": 497.00, "AXON": 293.95, "ACN": 318.00,
    "CRM": 318.00, "ADBE": 435.00, "YUM": 137.08, "NDAQ": 86.94,
    "AXP": 285.00, "CB": 301.00, "JPM": 237.50, "CDNS": 223.00,
    "GE": 208.92, "TSM": 317.00, "SNPS": 317.00, "LRCX": 211.00,
}

DEFENSIVE = ["NVO","JNJ","REGN","RMD","BSX","CHE","PEP","MCD",
             "NOC","HII","GD","ADP","UNP","WM","DHR","BR"]
GROWTH = ["LLY","META","SPGI","CPRT","POOL","HEI","TYL","VEEV",
          "DSGX","CBOE","DPZ","ODFL","TT","AME","MA","ICE"]

# Kings whose scanner checks fail for documented ACCOUNTING reasons
GREAT_OVERRIDES = {
    "NVO": "manual override - scanner disc 95.9% is DKK/USD artifact; real P/E ~14x; current-ratio + growth fails audited as structural/forward-looking",
    "MCD": "manual override - ROE fail = negative book equity from decades of buybacks (accounting artifact); sales/CFO multi-window warns = deliberate refranchising (higher-margin, era-documented); Adam watchlists it",
    "DPZ": "manual override - ROE fail = negative book equity (leveraged-buyback capital structure, by design); scanner IV $3,690 is the artifact, Adam's $421 adopted",
}

MOATS = {
    "NVO": "GLP-1 duopoly with LLY; insulin century-long franchise",
    "JNJ": "King of diversified healthcare; AAA balance sheet; 60y dividends",
    "REGN": "King of biologics discovery (VelocImmune); Eylea/Dupixent",
    "RMD": "King of sleep apnea (Philips still crippled); mask razor-blade",
    "BSX": "King of electrophysiology (Farapulse); cardio duopolies",
    "CHE": "King of hospice (VITAS) + Roto-Rooter monopoly",
    "PEP": "King of salty snacks (Frito-Lay); DSD distribution machine",
    "MCD": "King of restaurants, full stop; real-estate + franchise royalty",
    "NOC": "King of bombers (B-21) + nuclear triad lock-in",
    "HII": "ONLY builder of US nuclear carriers - literal monopoly",
    "GD": "King of business jets (Gulfstream) + submarines",
    "ADP": "King of payroll; highest switching costs in software-services",
    "UNP": "King of western rail; irreplaceable right-of-way",
    "WM": "King of waste; landfill permits cannot be replicated",
    "DHR": "King of bioprocessing; razor-blade consumables + DBS system",
    "BR": "King of proxy/shareholder comms - regulated monopoly",
    "LLY": "King of pharma; GLP-1 lead (tirzepatide) + Alzheimer's",
    "META": "King of social; 3.4B-user network effect",
    "SPGI": "King of ratings + owns the S&P 500 index itself",
    "CPRT": "King of salvage auctions; land + insurer relationships",
    "POOL": "King of pool distribution; 120k contractor relationships",
    "HEI": "King of PMA aircraft parts; FAA-approval moat",
    "TYL": "King of local-gov software; decades-long switching costs",
    "VEEV": "King of life-sciences SaaS; regulatory-validation lock-in",
    "DSGX": "King of logistics networks; 90% recurring",
    "CBOE": "King of volatility - SPX/VIX options monopoly",
    "DPZ": "King of pizza; franchise royalty + tech-first delivery",
    "ODFL": "King of LTL trucking; best OR in industry, decades running",
    "TT": "King of commercial HVAC (Trane); service annuity",
    "AME": "King of niche instruments; AMETEK roll-up discipline",
    "MA": "Payments duopoly co-king; toll booth on global commerce",
    "ICE": "King of energy futures + exchange data (NYSE owner)",
}

d = json.load(open(SCAN))
rows = {r["ticker"]: r for r in d["results"]}

def entry(t):
    r = rows.get(t)
    if r is None:
        print(f"WARN: {t} not in scan", file=sys.stderr); return None
    m = r["metrics"]
    g = m.get("proj_eps_next_5y")
    price = m.get("price")
    if t in ADAM_IV:
        iv = ADAM_IV[t]
        disc = round((iv - price) / iv * 100, 1) if price else None
        src = "Adam watchlist (base IV)"
    else:
        iv = m.get("intrinsic_value")
        disc = round(m["discount_pct"], 1) if m.get("discount_pct") is not None else None
        src = "scanner DCF"
    if t == "NVO":
        g, price, iv, disc, src = -0.92, 50.98, None, None, "manual (DKK artifact)"
    strat = "PMCC" if (g is not None and g <= 15) else "shares"
    e = {"ticker": t, "strategy": strat,
         "proj_growth_pct": round(g, 2) if g is not None else None,
         "price": price, "intrinsic_value": iv, "discount_pct": disc,
         "iv_source": src, "moat": MOATS.get(t, "")}
    if not r.get("is_great") and t not in GREAT_OVERRIDES:
        print(f"WARN: {t} not great, no override", file=sys.stderr)
    if t in GREAT_OVERRIDES:
        e["note"] = GREAT_OVERRIDES[t]
    if disc is not None and disc <= 0:
        print(f"WARN: {t} not below IV ({disc})", file=sys.stderr)
    return e

WATCH_ONLY = {
    "ISRG": "King of robotic surgery - but above Adam's IV (269.94 vs ~350)",
    "COST": "King of retail - above IV (576 vs ~949)",
    "V": "King of payments - above scanner IV (~344 vs 366)",
    "CTAS": "King of uniforms - above IV",
    "MCO": "Ratings co-king - just above Adam's IV (467 vs 476); first-pullback buy",
    "BRK-B": "The king - just above Adam's IV (497 vs 509); first-pullback buy",
    "MNST": "Energy-drink king - above Adam's IV (62 vs 97); scanner IV $1,209 was artifact",
    "AXON": "Taser/bodycam monopoly - far above Adam's IV (294 vs 526)",
    "IDXX": "King of vet diagnostics (the true ZTS upgrade) - above IV + ROE warn",
    "MSFT": "Above IV on fresh scan (447 vs 440)",
    "VRTX": "Near-miss this scan (FCF/ROE warns)",
    "TMO": "Near-miss this scan",
    "NFLX": "Near-miss this scan (CFO/FCF warns)",
    "VRSK": "NI warn but 45% disc - first re-entry when warn clears",
    "MELI": "Near-miss: ROIC history only 5y",
    "GOOG": "Scanner IV bug this scan - do not trust",
}

YOUR_CALL = {
    "ACN": "King of IT consulting; 49.3% below Adam's own IV (318 vs 161.25) - but same AI-disruption question you vetoed ADBE/INTU for. Your call.",
    "CRM": "King of CRM software; 43.9% below Adam's IV (318 vs 178.42) - but fails 3 scanner quality checks (ROE/ROIC/current ratio) AND carries the AI-disruption question. Weakest case.",
}

RESERVES = ["FDS","SHW","LIN","TJX","SAP","AOS","TTC","GGG","PAYX","SNA",
            "NKE","CTVA","LHX","CRH","NDSN","IEX","WSO","VMC","ITW","PCAR"]
DEMOTED = {
    "ZTS": "user veto - not best-of-breed (IDXX is the vet king, watch-only)",
    "CMG": "user veto - MCD is the restaurant king and took the slot",
    "AZN": "quality, but LLY/NVO are the pharma kings",
    "MDT": "largest medtech but not king of anything specific (ISRG/BSX are)",
    "AJG": "best-growing broker but MMC is the brokerage king",
    "TMUS": "best US carrier but carrier economics are not king-tier",
    "ADI": "analog #2 - TXN is the king (and both are AI-adjacent)",
    "ULTA": "category leader, not roi des rois vs COST/TJX",
    "CME": "futures king BUT above Adam's IV (221 vs 268) - watch, not buy",
    "FDS": "quality data vendor but Bloomberg is the king - reserve only",
}

def reserve(t):
    r = rows.get(t)
    m = r["metrics"] if r else {}
    price = m.get("price")
    if t in ADAM_IV and price:
        disc = round((ADAM_IV[t] - price) / ADAM_IV[t] * 100, 1)
    else:
        disc = round(m["discount_pct"], 1) if m.get("discount_pct") is not None else None
    return {"ticker": t, "discount_pct": disc}

payload = {
    "generated": "2026-07-31",
    "revision": "rev3-roi-des-rois",
    "scan": "fresh no-cache 2026-07-31, 540-ticker universe, 151 clean greats; Adam's watchlist IVs override scanner DCF where provided",
    "rules": "2000-account structure: max 6.25% of current account value per name, 3 tranches, adds at 200d SMA x1.01 while under IV and >=56d apart, never trim. PMCC (delta-0.80 LEAPS, sell delta-0.42 monthlies, roll at delta-0.80) if proj growth <=15%; plain shares if >15%.",
    "iv_divergences": {
        "MCD": "scanner IV $776 vs Adam $273 - scanner broken by negative book equity (buybacks); Adam adopted",
        "DPZ": "scanner IV $3,690 vs Adam $421 - same negative-equity artifact; Adam adopted",
        "PEP": "scanner IV $335 (58% disc) vs Adam $158 (11.8% disc) - scanner terminal multiple too generous on 5% grower; Adam adopted",
        "CME": "scanner IV $403 (33.5% disc) vs Adam $221 (ABOVE IV) - Adam adopted, CME demoted to watch",
        "FDS": "scanner IV $305 vs Adam $334 - close, both below price... both show discount; Adam adopted",
        "MNST": "scanner IV $1,209 = artifact; Adam $62 = above price - watch only",
        "ACN": "scanner IV $751 = artifact; Adam $318 = 49% discount - real but AI-disruption flagged",
    },
    "defensive_book": [e for e in (entry(t) for t in DEFENSIVE) if e],
    "growth_book": [e for e in (entry(t) for t in GROWTH) if e],
    "nvo_audit": ("Scanner fails audited: (1) decade-long current ratio 0.74-0.94 = structural rebate "
                  "liabilities, benign at 53-79% ROE; (2) analyst -0.92%/yr = real GLP-1-competition view; "
                  "(3) 95.9% scanner discount = DKK/USD artifact; manual P/E ~14x. Include as PMCC."),
    "watch_only": WATCH_ONLY,
    "your_call_ai_question": YOUR_CALL,
    "reserves_below_iv": [reserve(t) for t in RESERVES],
    "demoted_not_kings": DEMOTED,
    "excluded": {
        "ai_bubble_per_pyramid": ["NVDA","AVGO","ASML","TSM","MU","TER","NXPI","TXN","KEYS","TEL","FIX","EME","JCI","CDNS","SNPS","LRCX","GE"],
        "user_veto_ai_disruption": ["ADBE","INTU"],
        "artifact_iv_disc_gt_75": ["BF-B","EQT","DECK","PHM","DHI","GOOG","GOOGL","MU","CHD","TSM","HSY","GLOB","DG","ED","CTSH","KMI","IT","AKAM"],
    },
}

json.dump(payload, open(OUT, "w"), indent=1)
for name in ("defensive_book", "growth_book"):
    book = payload[name]
    p = sum(1 for e in book if e["strategy"] == "PMCC")
    print(f"{name}: {len(book)} names, {p} PMCC / {len(book)-p} shares")
print("wrote", OUT)
