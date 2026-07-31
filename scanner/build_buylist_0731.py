#!/usr/bin/env python3
"""Rebuild backtest/buylist_2026-07-31.json from fresh scan data.

Two 16-name books structured exactly like the 2000 account:
6.25% cap of current account value, 3 tranches, adds at 200d SMA x1.01
under IV >= 56d apart, never trim. PMCC if g<=15%, plain shares if g>15%.
NVO is a hardcoded manual override (DKK/USD IV artifact, real P/E ~14x).
"""
import json, os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN = os.path.join(BASE, "public", "data", "scan_results.json")
OUT = os.path.join(BASE, "backtest", "buylist_2026-07-31.json")

DEFENSIVE = ["ZTS","RMD","REGN","JNJ","CHE","AZN","MDT","BSX","NVO","PEP","NOC","HII","GD","CME","ADP","UNP"]
GROWTH = ["META","TMUS","VEEV","BR","TYL","DSGX","HEI","AJG","CPRT","POOL","CBOE","ULTA","CMG","LLY","ADI","SPGI"]
WATCH_ONLY = {
    "MSFT": "Flipped above intrinsic value on fresh scan (447 vs 440) - wait for pullback",
    "VRTX": "Slipped to near-miss on fresh data (FCF/ROE warns) - re-check next scan",
    "TMO": "Slipped to near-miss on fresh data - re-check next scan",
    "NFLX": "Slipped to near-miss (CFO/FCF warns) - re-check next scan",
    "VRSK": "NI warn but 45% discount - first re-entry candidate when warn clears",
    "MELI": "Near-miss: ROIC history only 5y (young float) - quality is real, data too short",
    "GOOG": "IV = $8,276 this scan = projection data bug - do not trust discount",
}
RESERVES = ["AOS","TTC","GGG","PAYX","SNA","NKE","CTVA","LHX","CRH","NDSN","IEX","DHR","WSO","VMC","ITW","PCAR","FDS","SAP","ICE","TJX","MA"]

MOATS = {
    "ZTS": "Pet-med #1; vets prescribe by brand; R&D scale (sanity-check 70% disc before sizing)",
    "RMD": "Sleep-apnea duopoly (vs Philips, still crippled); mask razor-blade model",
    "REGN": "Eylea/Dupixent biologics; VelocImmune discovery engine",
    "JNJ": "Diversified pharma+medtech; AAA-grade balance sheet; 60y dividend record",
    "CHE": "VITAS hospice + Roto-Rooter - two boring monopolies in one ticker",
    "AZN": "Oncology pipeline depth; EM distribution few peers match",
    "MDT": "Largest pure-play medtech; surgeon training lock-in",
    "BSX": "Cardio device duopolies; category killer in electrophysiology",
    "NVO": "GLP-1 duopoly with LLY; insulin century-long franchise; DKK cost base",
    "PEP": "Snacks (Frito-Lay) is the moat, not soda; DSD distribution machine",
    "NOC": "B-21 + nuclear triad prime; classified-program lock-in",
    "HII": "ONLY builder of US nuclear carriers - literal monopoly",
    "GD": "Gulfstream + submarines; backlog measured in decades",
    "CME": "Rates/commodities futures monopoly; clearing network effect",
    "ADP": "Payroll = highest switching-cost SaaS before SaaS existed",
    "UNP": "Western rail duopoly; irreplaceable right-of-way",
    "META": "3.4B user network effect; ad auction scale",
    "TMUS": "Best 5G spectrum position; churn lowest in industry",
    "BR": "Proxy/shareholder-communications regulated monopoly - pure financial toll booth",
    "TYL": "Local-gov software; decades-long switching costs",
    "DSGX": "Logistics network effect; 90% recurring revenue",
    "HEI": "FAA-approval moat on aircraft parts aftermarket",
    "AJG": "Insurance brokerage scale + tuck-in M&A machine",
    "CPRT": "Salvage-auction duopoly; land + insurer relationships",
    "POOL": "Pool-supply distribution monopoly; 120k contractor relationships",
    "CBOE": "Monopoly on SPX/VIX options - the volatility toll booth",
    "ULTA": "Beauty category killer; loyalty program 44M members",
    "CMG": "Only scaled fast-casual with food-with-integrity brand",
    "LLY": "GLP-1 duopoly (tirzepatide leads on efficacy); Alzheimer's optionality",
    "ADI": "Analog design moat - 10y+ product lives, sticky sockets (allowed AI-adjacent quota name)",
    "SPGI": "Ratings duopoly + index licensing (S&P 500 itself)",
    "AOS": "Water-heater brand + replacement cycle", "TTC": "Turf-care brand + dealer network",
    "GGG": "Fluid-handling niches, pricing power",
    "PAYX": "SMB payroll switching costs", "SNA": "Brand + franchisee van network + financing",
    "NKE": "Global athletic brand (turnaround watch)", "CTVA": "Seed/crop-protection duopoly",
    "LHX": "Defense comms prime", "CRH": "Aggregates local monopolies",
    "NDSN": "Precision dispensing niches", "IEX": "Niche fluidics roll-up",
    "DHR": "Bioprocessing razor-blade", "WSO": "HVAC distribution scale",
    "VMC": "Aggregates - quarries can't be replicated", "ITW": "80/20 niche dominance",
    "PCAR": "Truck brand + parts annuity", "FDS": "Financial data sticky terminals",
    "SAP": "ERP switching costs", "ICE": "Exchange + mortgage-data toll booth",
    "TJX": "Off-price treasure-hunt scale", "MA": "Payments duopoly toll booth",
    "VEEV": "Healthcare-vertical SaaS monopoly (CRM/Vault); regulatory-validation lock-in resists AI disruption",
}

d = json.load(open(SCAN))
rows = {r["ticker"]: r for r in d["results"]}

def entry(t):
    if t == "NVO":
        return {"ticker": "NVO", "strategy": "PMCC", "proj_growth_pct": -0.92,
                "price": 50.98, "intrinsic_value": None, "discount_pct": None,
                "moat": MOATS["NVO"],
                "note": "manual override - scanner disc 95.9% is DKK/USD artifact; real P/E ~14x"}
    r = rows.get(t)
    if r is None:
        print(f"WARN: {t} not in scan results", file=sys.stderr); return None
    m = r["metrics"]
    g = m.get("proj_eps_next_5y")
    strat = "PMCC" if (g is not None and g <= 15) else "shares"
    if not r.get("is_great"):
        print(f"WARN: {t} not flagged great this scan", file=sys.stderr)
    disc = m.get("discount_pct")
    if disc is not None and disc <= 0:
        print(f"WARN: {t} not below IV (disc {disc})", file=sys.stderr)
    return {"ticker": t, "strategy": strat,
            "proj_growth_pct": round(g, 2) if g is not None else None,
            "price": m.get("price"), "intrinsic_value": m.get("intrinsic_value"),
            "discount_pct": round(disc, 1) if disc is not None else None,
            "moat": MOATS.get(t, "")}

def reserve(t):
    r = rows.get(t)
    if r is None: return {"ticker": t, "moat": MOATS.get(t, "")}
    m = r["metrics"]
    disc = m.get("discount_pct")
    return {"ticker": t, "discount_pct": round(disc, 1) if disc is not None else None,
            "moat": MOATS.get(t, "")}

payload = {
    "generated": "2026-07-31",
    "scan": "fresh no-cache 2026-07-31, 540-ticker universe (S&P500 + Dow30 + NDX + curated extras), 151 clean greats, 117 below IV",
    "rules": "2000-account structure: max 6.25% of current account value per name, 3 tranches, adds at 200d SMA x1.01 while under IV and >=56d apart, never trim. PMCC (delta-0.80 LEAPS, sell delta-0.42 monthlies, roll at delta-0.80) if proj growth <=15%; plain shares if >15%.",
    "defensive_book": [e for e in (entry(t) for t in DEFENSIVE) if e],
    "growth_book": [e for e in (entry(t) for t in GROWTH) if e],
    "nvo_audit": ("Scanner fails audited: (1) current ratio 0.74-0.94 for a decade = structural rebate liabilities, "
                  "benign at 53-79% ROE - not short-term; (2) analyst LT growth -0.92%/yr is a real GLP-1-competition "
                  "forward view vs +16.7%/yr actual 2019-25 revenue CAGR - real, not noise; (3) the 95.9% scanner "
                  "discount is a DKK-financials vs USD-ADR currency artifact - manual math: EPS ~23 DKK = ~$3.60, "
                  "P/E ~14x. Verdict: include as PMCC (cap the upside you're least sure of)."),
    "watch_only": WATCH_ONLY,
    "reserves_below_iv": [reserve(t) for t in RESERVES],
    "excluded": {
        "ai_bubble_per_pyramid": ["NVDA","AVGO","ASML","TSM","MU","TER","NXPI","TXN","KEYS","TEL","FIX","EME","JCI"],
        "artifact_iv_disc_gt_75": ["BF-B","EQT","DECK","PHM","DHI","GOOG","GOOGL","MU","CHD","TSM","HSY","GLOB","DG","ED","CTSH","ACN","KMI","IT","AKAM"],
        "user_veto_ai_disruption": ["ADBE","INTU"],
    },
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(payload, open(OUT, "w"), indent=1)
db = payload["defensive_book"]; gb = payload["growth_book"]
for name, book in (("defensive_book", db), ("growth_book", gb)):
    p = sum(1 for e in book if e["strategy"] == "PMCC")
    print(f"{name}: {len(book)} names, {p} PMCC / {len(book)-p} shares")
print("wrote", OUT)
