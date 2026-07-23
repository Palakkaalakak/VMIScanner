"""Curated extras universe — quality compounders outside S&P500/Dow/NDX.

Added per user instruction ("Why are MELI, POOL, CNSWF, DSGX, NVO and
quite a few other great companies not in scan? Expand to include their
universes."). Index membership is an accident of committee rules —
foreign ADRs (NVO, ASML), Toronto listings traded OTC in the US (CNSWF),
Canadian SEC registrants (DSGX), and ex-constituents (POOL, MKTX) are
all investable wide-moat candidates the index universes miss.

This is a STATIC watchlist by design: every name still goes through the
exact same deep fundamental checks + DCF as index constituents. Being
listed here grants a scan, not a pass.

Data-source note: NVO/ASML/TSM etc. file 20-F/6-K with the SEC so
Company Facts usually works; CNSWF (SEDAR filer) falls through to the
Yahoo source; that fallback order already exists in scan.py.
"""
from typing import Dict, List

# ticker -> (company, sector, industry)
EXTRA_TICKERS = {
    # --- ex-index / never-index US quality ---
    "POOL":  ("Pool Corporation", "Consumer Discretionary", "Distributors"),
    "MKTX":  ("MarketAxess Holdings", "Financials", "Financial Exchanges & Data"),
    "WSO":   ("Watsco", "Industrials", "Trading Companies & Distributors"),
    "HEI":   ("HEICO Corporation", "Industrials", "Aerospace & Defense"),
    "ROL":   ("Rollins", "Industrials", "Environmental & Facilities Services"),
    "TTC":   ("Toro Company", "Industrials", "Agricultural & Farm Machinery"),
    "GGG":   ("Graco", "Industrials", "Industrial Machinery"),
    "RLI":   ("RLI Corp", "Financials", "Property & Casualty Insurance"),
    "CHE":   ("Chemed", "Health Care", "Health Care Services"),
    # --- foreign ADRs / US-listed foreign quality ---
    "NVO":   ("Novo Nordisk (ADR)", "Health Care", "Pharmaceuticals"),
    "ASML":  ("ASML Holding (ADR)", "Information Technology", "Semiconductor Equipment"),
    "TSM":   ("Taiwan Semiconductor (ADR)", "Information Technology", "Semiconductors"),
    "NVS":   ("Novartis (ADR)", "Health Care", "Pharmaceuticals"),
    "AZN":   ("AstraZeneca (ADR)", "Health Care", "Pharmaceuticals"),
    "SAP":   ("SAP SE (ADR)", "Information Technology", "Application Software"),
    "UL":    ("Unilever (ADR)", "Consumer Staples", "Personal Care Products"),
    "NSRGY": ("Nestle (ADR, OTC)", "Consumer Staples", "Packaged Foods & Meats"),
    "LVMUY": ("LVMH (ADR, OTC)", "Consumer Discretionary", "Apparel, Accessories & Luxury Goods"),
    "LRLCY": ("L'Oreal (ADR, OTC)", "Consumer Staples", "Personal Care Products"),
    "ATLKY": ("Atlas Copco (ADR, OTC)", "Industrials", "Industrial Machinery"),
    # --- Canadian quality (US listings / OTC) ---
    "CNSWF": ("Constellation Software (OTC)", "Information Technology", "Application Software"),
    "DSGX":  ("Descartes Systems Group", "Information Technology", "Application Software"),
    "TRI":   ("Thomson Reuters", "Industrials", "Research & Consulting Services"),
    "WCN":   ("Waste Connections", "Industrials", "Environmental & Facilities Services"),
    # --- LatAm / other growth listed on US exchanges ---
    "MELI":  ("MercadoLibre", "Consumer Discretionary", "Broadline Retail"),
    "GLOB":  ("Globant", "Information Technology", "IT Consulting & Other Services"),
}


def fetch_extras(use_cache: bool = True) -> List[Dict]:  # noqa: ARG001 — seam parity
    """Return the curated extras as universe rows (no network call)."""
    return [{
        "ticker": t,
        "company": c,
        "sector": s,
        "sub_industry": ind,
        "industry": ind,
        "country": "",
        "market_cap": "",
    } for t, (c, s, ind) in EXTRA_TICKERS.items()]


if __name__ == "__main__":
    rows = fetch_extras()
    print(f"{len(rows)} curated extras")
    for r in rows[:5]:
        print(r)
