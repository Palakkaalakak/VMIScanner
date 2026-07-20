"""Build 16-stock GROWTH and DEFENSIVE books per vintage from the PIT scan.

Selection (per user instruction):
  1. Universe = scanner-verified GREAT businesses at the vintage (PIT data
     only — SEC fundamentals truncated to fiscal years ending pre-vintage).
  2. MANUAL wide-moat pass using Adam's framework (Lesson 4 / E1), judged
     with ONLY era-knowable information: moat must come from brand monopoly,
     network effect, high switching costs, high barriers to entry, or huge
     economies of scale. Technological innovation, standard patents and
     pharma patents are NOT sustainable moats. Commodity businesses
     (oil & gas, steel, memory/drives, chemicals, homebuilders) = no moat.
  3. ANTI-BUBBLE portion (user requirement; the 2000 vintage was wholly
     anti-bubble by avoiding dotcom): each book must draw at least HALF its
     names from era-reasonably-priced stocks (trailing PE <= 25 at the
     vintage — a valuation fact knowable at the time, not hindsight).
     Era-identified froth is also excluded in the moat pass itself:
       * 2015: the then-widely-discussed biotech bubble (pharma/biotech
         momentum names are already out as patent-moats).
       * 2020: late-cycle high-multiple SaaS/momentum tech — no name enters
         on growth alone if its era PE marks it as bubble-priced (>50).
  4. GROWTH book = 8 highest-growth wide-moats priced PE<=25 (anti-bubble
     half) + 8 highest-growth remaining wide-moats with PE<=50.
     DEFENSIVE book = 16 lowest-beta wide-moats not in the growth book,
     with the same >=8 anti-bubble (PE<=25) requirement.
DCF inputs all point-in-time data-derived (era PE / g / beta / CFO-NI mult).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- manual wide-moat calls (ticker -> moat source, era-knowable only) ----
WIDE_2020 = {
    "MA": "network effect (payments duopoly)",
    "GPN": "switching costs (merchant processing integration)",
    "CBOE": "network effect + regulation (options exchange)",
    "ICE": "network effect + regulation (NYSE/data)",
    "FISV": "switching costs (bank core processing)",
    "NDAQ": "network effect + regulation (exchange)",
    "JKHY": "switching costs (bank core processing)",
    "PYPL": "network effect (two-sided payments)",
    "MSFT": "switching costs + network effect (Windows/Office/Azure)",
    "INTU": "switching costs (QuickBooks/TurboTax)",
    "CDNS": "switching costs (EDA duopoly)",
    "SNPS": "switching costs (EDA duopoly)",
    "ANET": "switching costs (cloud networking EOS)",
    "NVDA": "ecosystem barriers (CUDA, known by 2020) + scale",
    "ADI": "switching costs (analog design-ins)",
    "TXN": "scale + switching costs (analog catalog)",
    "KLAC": "barriers (process-control near-monopoly)",
    "LRCX": "barriers (etch/deposition oligopoly)",
    "AMAT": "barriers (semi-equipment scale)",
    "GOOGL": "network effect (search/ads) — Adam wide-moat example",
    "AMZN": "scale + network effect — Adam wide-moat example",
    "ADP": "switching costs (payroll)",
    "PAYX": "switching costs (payroll)",
    "BR": "switching costs (shareholder comms near-monopoly)",
    "ACN": "brand + scale (embedded consulting)",
    "ISRG": "switching costs + barriers (da Vinci installed base)",
    "ZTS": "brand + scale (animal health leader)",
    "SYK": "switching costs (surgeon training/installed base)",
    "EW": "barriers (structural heart)",
    "MDT": "scale + switching costs (device breadth)",
    "TMO": "scale + switching costs (lab ecosystem)",
    "A": "switching costs (instruments installed base)",
    "MTD": "switching costs (lab weighing near-monopoly)",
    "RMD": "barriers (sleep apnea duopoly)",
    "UNH": "scale (largest US insurer network)",
    "COST": "scale + membership loyalty",
    "MNST": "brand (energy drinks, KO distribution)",
    "HSY": "brand monopoly (US chocolate)",
    "PEP": "brand + distribution scale",
    "MCD": "brand + real estate scale — Adam example",
    "NKE": "brand monopoly — Adam example",
    "TJX": "scale (off-price buying network)",
    "ROST": "scale (off-price)",
    "ORLY": "scale + distribution density (auto parts)",
    "DG": "scale + rural footprint density",
    "DLTR": "scale (extreme-value format)",
    "TSCO": "niche scale (rural lifestyle retail)",
    "ULTA": "scale + loyalty (beauty specialty)",
    "CHD": "brand portfolio",
    "EL": "brand portfolio (prestige beauty)",
    "HRL": "brand portfolio (protein)",
    "SYY": "scale + distribution density (foodservice)",
    "WM": "regulation + scale (landfill permits)",
    "SHW": "brand + distribution density (paint)",
    "ECL": "switching costs (embedded hygiene service)",
    "SPGI": "regulation + network effect (ratings) — Adam example",
    "CTAS": "scale + route density (uniforms)",
    "CPRT": "network effect (salvage auctions)",
    "FAST": "scale + vending/onsite embedment",
    "GWW": "scale (MRO distribution)",
    "ITW": "switching costs (engineered components)",
    "HON": "switching costs (aerospace/controls)",
    "NOC": "regulation + barriers (defense prime)",
    "GD": "regulation + barriers (defense prime)",
    "LHX": "regulation + barriers (defense electronics)",
    "APH": "switching costs (connector design-ins)",
    "ROP": "switching costs (niche roll-up)",
    "AME": "switching costs (niche instruments)",
    "IEX": "switching costs (niche fluidics)",
    "LLY": "scale + pipeline breadth",
    "AAPL": "brand + ecosystem switching costs — Adam example",
}
# 2020 exclusions (era-knowable reasoning): CHRW/EXPD freight brokerage narrow
# (CHRW 61% 'growth' = ASC-606 gross-revenue presentation artifact);
# VRTX/REGN/BIIB/AMGN/BMY/MRK pharma-patent moats (not sustainable per Adam);
# ALGN patents + competition; FTNT/FFIV/AKAM innovation moats; LEN/PHM
# homebuilders no-moat; NUE/MU/WDC/NTAP commodity; EOG/MPC/KMI/WMB oil&gas
# no-moat; CE/PKG/AVY/IFF/PPG/VMC/MLM materials narrow; JCI/BWA/TXT/SWK/AOS/
# XYL/SNA cyclicals narrow; GRMN/EA/HAS/BBY/AAP consumer narrow; UHS/LH/DGX/
# HSIC/COO/CAH/CVS health services narrow; GL/AON insurance narrow; OMC/VZ
# narrow; WAB/CMI/CAT/DE machinery narrow; CSX/NSC/JBHT rails/trucking
# excluded on cyclicality judgment; ILMN single-platform innovation moat;
# BKNG excluded on travel-cyclicality judgment; LKQ narrow; GPN PE86 and
# FISV PE68 remain in pool but the PE<=50 bubble gate keeps them out of books.

WIDE_2015 = {
    "AAPL": "brand + ecosystem switching costs — Adam example",
    "MSFT": "switching costs + network effect",
    "MA": "network effect (payments duopoly)",
    "BKNG": "network effect (OTA two-sided marketplace)",
    "MNST": "brand (energy drinks, KO distribution)",
    "COST": "scale + membership loyalty",
    "ISRG": "switching costs + barriers (da Vinci)",
    "EW": "barriers (structural heart)",
    "TMO": "scale + switching costs",
    "ROP": "switching costs (niche roll-up)",
    "AME": "switching costs (niche instruments)",
    "KLAC": "barriers (process control)",
    "FAST": "scale + onsite embedment",
    "TSCO": "niche scale (rural retail)",
    "DLTR": "scale (extreme value)",
    "DG": "scale + rural density",
    "ORLY": "scale + distribution density",
    "ROST": "scale (off-price)",
    "SHW": "brand + distribution density",
    "ECL": "switching costs (embedded service)",
    "MCD": "brand + real estate scale",
    "PEP": "brand + distribution scale",
    "CL": "brand monopoly (toothpaste)",
    "GIS": "brand portfolio",
    "MKC": "brand + scale (spices near-monopoly)",
    "HRL": "brand portfolio (protein)",
    "SJM": "brand portfolio",
    "HD": "scale + pro network density",
    "JNJ": "brand + scale breadth — Adam example",
    "BDX": "switching costs (consumables installed base)",
    "PAYX": "switching costs (payroll)",
    "FISV": "switching costs (bank core processing)",
    "AON": "scale + data (brokerage)",
    "MO": "brand + regulation (tobacco)",
    "BF.B": "brand monopoly (whiskey)",
    "EL": "brand portfolio (prestige beauty)",
    "QCOM": "barriers + licensing lock (mobile SoC/IP)",
    "INTU": "switching costs (QuickBooks/TurboTax)",
    "ACN": "brand + scale (consulting)",
    "CTSH": "switching costs (embedded IT outsourcing)",
    "APH": "switching costs (connector design-ins)",
    "UNH": "scale (insurer network)",
    "WAT": "switching costs (LC/MS installed base)",
    "UPS": "scale + network density",
    "CVS": "scale (pharmacy + PBM integration)",
}
# 2015 exclusions: ORCL g102% = revenue-tag artifact; GILD/BIIB/AMGN pharma-
# patent moats AND the well-publicized 2015 biotech bubble (anti-bubble);
# UAA/FOSL/URBN/RL/VFC fashion narrow/no moat; TRIP narrow; HAL/SLB/NOV/HP/
# KMI/MPC/EMN oil&gas/commodity; WDC/NTAP commodity hardware; SWK g31% =
# Black&Decker merger artifact; ETN/CMI/BWA/AN/UHS/AKAM/FFIV/KR narrow.

VINTAGES = {
    2020: {"wide": WIDE_2020, "date": "2020-01-02"},
    2015: {"wide": WIDE_2015, "date": "2015-01-02"},
}
PE_VALUE = 25.0   # anti-bubble slice: era-reasonable trailing multiple
PE_BUBBLE = 50.0  # bubble-priced gate: never enters a book


def build(year):
    cfg = VINTAGES[year]
    d = json.load(open(os.path.join(HERE, f"vintage_inputs_{year}.json")))
    cands = {c["ticker"]: c for c in d["candidates"]
             if c["trailing_pe"] and c["growth"] is not None
             and c["beta"] is not None and c["ocf_mult"]}
    wide = {t: c for t, c in cands.items() if t in cfg["wide"]}
    missing = [t for t in cfg["wide"] if t not in cands]

    def pick_growth(pool):
        value = sorted((c for c in pool if c["trailing_pe"] <= PE_VALUE),
                       key=lambda c: -c["growth"])[:8]
        chosen = {c["ticker"] for c in value}
        rest = sorted((c for c in pool if c["ticker"] not in chosen
                       and c["trailing_pe"] <= PE_BUBBLE),
                      key=lambda c: -c["growth"])[:16 - len(value)]
        return value + rest

    growth = pick_growth(list(wide.values()))
    gset = {c["ticker"] for c in growth}
    remaining = [c for c in wide.values() if c["ticker"] not in gset
                 and c["trailing_pe"] <= PE_BUBBLE]
    dvalue = sorted((c for c in remaining if c["trailing_pe"] <= PE_VALUE),
                    key=lambda c: c["beta"])[:8]
    dset = {c["ticker"] for c in dvalue}
    drest = sorted((c for c in remaining if c["ticker"] not in dset),
                   key=lambda c: c["beta"])[:8]
    defensive = sorted(dvalue + drest, key=lambda c: c["beta"])

    def book(rows):
        return {c["ticker"]: {
            "pe": c["trailing_pe"], "g": round(c["growth"], 4),
            "beta": c["beta"], "ocf_mult": c["ocf_mult"],
            "moat": cfg["wide"][c["ticker"]], "sector": c["sector"],
            "anti_bubble": c["trailing_pe"] <= PE_VALUE,
        } for c in rows}

    gb, db = book(growth), book(defensive)
    out = {"vintage": year, "date": cfg["date"], "rf": d["rf_10y"], "mrp": 0.04,
           "growth_book": gb, "defensive_book": db,
           "anti_bubble_count": {"growth": sum(v["anti_bubble"] for v in gb.values()),
                                  "defensive": sum(v["anti_bubble"] for v in db.values())},
           "wide_moat_pool": len(wide), "not_in_candidates": missing}
    json.dump(out, open(os.path.join(HERE, f"books_{year}.json"), "w"), indent=1)
    print(f"=== {year} (RF {d['rf_10y']*100:.2f}%) wide pool {len(wide)}; missing: {missing}")
    print("GROWTH :", " ".join(f"{t}(g{v['g']*100:.0f},PE{v['pe']:.0f}{'*' if v['anti_bubble'] else ''})"
                               for t, v in gb.items()))
    print("DEFENSE:", " ".join(f"{t}(b{v['beta']:.2f},PE{v['pe']:.0f}{'*' if v['anti_bubble'] else ''})"
                               for t, v in db.items()))
    print(f"anti-bubble(*=PE<=25): growth {out['anti_bubble_count']['growth']}/16, "
          f"defensive {out['anti_bubble_count']['defensive']}/16")
    return out


if __name__ == "__main__":
    for y in (2020, 2015):
        build(y)
