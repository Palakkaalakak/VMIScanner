"""Hand-built era growth books for deep vintages: 1990, 1995, 2005, 2010.

SEC XBRL only reaches ~2007+, so the PIT scanner cannot produce these
vintages (2010 scan returned 0 GREATs on 5y-history grounds). These books
are therefore hand-built the same way as the original 2000 book
(simulate2.py): era-documented trailing PEs and era-known 5y growth rates
(what an investor reading annual reports at the time would have used),
manual era wide-moat review per Adam's framework (brand / network effect /
switching costs / barriers / scale; no pharma-patent or commodity moats),
NO hindsight in selection. Betas are COMPUTED from 5y weekly returns vs
^GSPC ending at the vintage date (deep_betas.json). RF = FRED DGS10 at
vintage.

All books are GROWTH books (user: growth-heavy, no defensive) with the
25% sector cap (max 4 of 16 per GICS sector). Anti-bubble slice: >= 8/16
at era trailing PE <= 25; nothing above PE 50 (ISRG at 2010, era PE ~65,
was excluded by this gate despite being an era wide-moat).

Scandal-sell events (same standard as the 2000 run: accounting fraud or
criminal probe of the core business, judged with era information only):
  * UNH options-backdating scandal, Oct 2006  -> sell (1995, 2005 books)
  * WMT Mexico FCPA bribery + cover-up probe, Apr 2012 -> sell (all books
    holding WMT: 1990, 1995, 2005, 2010)
  * UNH DOJ criminal probe (Medicare billing), May 2025 -> sell (2010 book)
Replacements = era wide-moat brand staples from the replacement pool with
data coverage (CL, GIS, CHD), chosen by era rank, not performance.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BETAS = json.load(open(os.path.join(HERE, "deep_betas.json")))


def _beta(t, vd, years=5):
    """5y weekly beta vs ^GSPC ending at vintage date (aligned grid)."""
    import numpy as np
    import pandas as pd
    px = pd.read_csv(os.path.join(HERE, "weekly_deep.csv"),
                     index_col=0, parse_dates=True)
    W = px.resample("W-FRI").last()
    end = pd.Timestamp(vd)
    start = end - pd.DateOffset(years=years)
    df = pd.concat([W[t].loc[start:end].pct_change(fill_method=None),
                    W["^GSPC"].loc[start:end].pct_change(fill_method=None)],
                   axis=1).dropna()
    return round(float(np.cov(df.iloc[:, 0], df.iloc[:, 1])[0, 1]
                       / np.var(df.iloc[:, 1])), 2)

# ticker: (era trailing PE, era-known growth, sector, moat note, ocf_mult)
B1990 = {
    "WMT":  (29, .25, "Consumer Staples", "scale + distribution (everyday low cost)", 1.15),
    "HD":   (33, .30, "Consumer Discretionary", "scale + category-killer barriers", 1.20),
    "DIS":  (23, .20, "Consumer Discretionary", "brand monopoly (characters/parks)", 1.30),
    "NKE":  (13, .20, "Consumer Discretionary", "brand (athletic footwear)", 1.10),
    "MCD":  (15, .12, "Consumer Discretionary", "brand + real estate + franchising", 1.50),
    "KO":   (18, .12, "Consumer Staples", "brand monopoly + distribution", 1.20),
    "PEP":  (18, .15, "Consumer Staples", "brand + snacks distribution (Frito-Lay)", 1.25),
    "SYY":  (19, .15, "Consumer Staples", "scale (foodservice distribution)", 1.20),
    "JNJ":  (16, .12, "Health Care", "brand + diversified scale", 1.15),
    "ABT":  (16, .14, "Health Care", "diversified scale (diagnostics/nutrition)", 1.20),
    "MDT":  (15, .13, "Health Care", "switching costs (implantables)", 1.20),
    "ADP":  (19, .13, "Industrials", "switching costs (payroll processing)", 1.25),
    "GWW":  (15, .11, "Industrials", "scale + distribution density (MRO)", 1.15),
    "EMR":  (13, .10, "Industrials", "switching costs (process automation)", 1.20),
    "ITW":  (14, .12, "Industrials", "niche barriers (engineered fasteners)", 1.20),
    "AXP":  (11, .11, "Financials", "brand + closed-loop network", 1.20),
}

B1995 = {
    "MSFT": (25, .30, "Information Technology", "switching costs + network (Windows/Office)", 1.25),
    "INTC": (10, .30, "Information Technology", "scale + barriers (x86 near-monopoly)", 1.35),
    "ORCL": (30, .30, "Information Technology", "switching costs (enterprise DB)", 1.20),
    "UNH":  (19, .35, "Health Care", "scale (managed-care network)", 1.30),
    "HD":   (24, .25, "Consumer Discretionary", "scale + category-killer barriers", 1.15),
    "NKE":  (13, .15, "Consumer Discretionary", "brand (athletic)", 1.10),
    "DIS":  (21, .15, "Consumer Discretionary", "brand monopoly", 1.30),
    "MCD":  (16, .12, "Consumer Discretionary", "brand + franchising", 1.50),
    "WMT":  (19, .20, "Consumer Staples", "scale + distribution", 1.20),
    "KO":   (22, .13, "Consumer Staples", "brand monopoly", 1.20),
    "SYY":  (18, .14, "Consumer Staples", "scale (foodservice)", 1.20),
    "JNJ":  (16, .11, "Health Care", "brand + diversified scale", 1.15),
    "MDT":  (19, .15, "Health Care", "switching costs (implantables)", 1.20),
    "ADP":  (20, .13, "Industrials", "switching costs (payroll)", 1.25),
    "GWW":  (15, .10, "Industrials", "scale (MRO distribution)", 1.15),
    "AXP":  (11, .10, "Financials", "brand + closed-loop network", 1.20),
}

B2005 = {
    "QCOM": (30, .20, "Information Technology", "licensing lock + barriers (CDMA IP)", 1.30),
    "MSFT": (22, .12, "Information Technology", "switching costs + network", 1.30),
    "UNH":  (19, .25, "Health Care", "scale (largest managed-care network)", 1.30),
    "SYK":  (29, .20, "Health Care", "switching costs (orthopedics)", 1.20),
    "BDX":  (19, .10, "Health Care", "scale + switching costs (med consumables)", 1.35),
    "JNJ":  (21, .12, "Health Care", "brand + diversified scale", 1.15),
    "LOW":  (21, .18, "Consumer Discretionary", "scale duopoly (home improvement)", 1.20),
    "HD":   (19, .12, "Consumer Discretionary", "scale duopoly", 1.20),
    "TJX":  (16, .12, "Consumer Discretionary", "scale + buying network (off-price)", 1.30),
    "ROST": (17, .14, "Consumer Discretionary", "scale (off-price)", 1.25),
    "WMT":  (21, .11, "Consumer Staples", "scale + distribution", 1.25),
    "SYY":  (24, .12, "Consumer Staples", "scale (foodservice)", 1.20),
    "MO":   (13, .08, "Consumer Staples", "brand monopoly + pricing power", 1.15),
    "DHR":  (22, .15, "Industrials", "DBS system + niche instrument barriers", 1.30),
    "UPS":  (29, .10, "Industrials", "network density (parcel duopoly)", 1.40),
    "AXP":  (19, .12, "Financials", "brand + closed-loop network", 1.20),
}

B2010 = {
    "AAPL":  (32, .35, "Information Technology", "brand + ecosystem switching costs", 1.30),
    "GOOGL": (38, .30, "Information Technology", "network effect (search near-monopoly)", 1.35),
    "MSFT":  (19, .10, "Information Technology", "switching costs + network", 1.30),
    "QCOM":  (28, .12, "Information Technology", "licensing lock (3G IP)", 1.30),
    "SYK":   (17, .12, "Health Care", "switching costs (orthopedics)", 1.20),
    "UNH":   (9,  .10, "Health Care", "scale (managed-care network)", 1.30),
    "BDX":   (14, .10, "Health Care", "scale + switching costs", 1.35),
    "TJX":   (15, .10, "Consumer Discretionary", "scale + buying network (off-price)", 1.30),
    "ROST":  (15, .12, "Consumer Discretionary", "scale (off-price)", 1.25),
    "MCD":   (16, .08, "Consumer Discretionary", "brand + franchising + real estate", 1.45),
    "NKE":   (19, .08, "Consumer Discretionary", "brand (athletic)", 1.15),
    "WMT":   (14, .08, "Consumer Staples", "scale + distribution", 1.30),
    "UNP":   (16, .10, "Industrials", "barriers (irreplaceable rail network)", 1.60),
    "FAST":  (29, .12, "Industrials", "distribution density + onsite embedment", 1.10),
    "DHR":   (21, .12, "Industrials", "DBS + niche instrument barriers", 1.30),
    "ADP":   (15, .08, "Industrials", "switching costs (payroll)", 1.25),
}

RF = {1990: .0794, 1995: .0788, 2005: .0423, 2010: .0385}
DATES = {1990: "1990-01-02", 1995: "1995-01-03",
         2005: "2005-01-03", 2010: "2010-01-04"}
BOOKS = {1990: B1990, 1995: B1995, 2005: B2005, 2010: B2010}

# replacement params at their scandal entry (era-documented PE/g,
# computed beta at entry date from deep_betas.json)
REPL_PARAMS = {
    ("CL", "2006-10-16"):  (24, .10, BETAS["2006-10-16"]["CL"], 1.30),
    ("GIS", "2012-04-23"): (15, .08, 0.32, 1.20),
    ("CHD", "2025-05-19"): (24, .08, BETAS["2025-05-19"]["CHD"], 1.20),
}


def main():
    from collections import Counter
    for year, bk in BOOKS.items():
        vd = DATES[year]
        secs = Counter(v[2] for v in bk.values())
        assert max(secs.values()) <= 4, f"{year} sector cap violated: {secs}"
        ab = sum(1 for v in bk.values() if v[0] <= 25)
        assert ab >= 8, f"{year} anti-bubble {ab}/16"
        assert all(v[0] <= 50 for v in bk.values())
        book = {}
        for t, (pe, g, sec, moat, m) in bk.items():
            beta = BETAS.get(vd, {}).get(t)
            if beta is None:
                beta = _beta(t, vd)
                BETAS.setdefault(vd, {})[t] = beta
            book[t] = {"pe": pe, "g": g, "beta": beta, "ocf_mult": m,
                       "moat": moat, "sector": sec, "anti_bubble": pe <= 25}
        out = {"vintage": year, "date": vd, "rf": RF[year], "mrp": .04,
               "sector_cap": 4, "growth_book": book,
               "anti_bubble_count": ab, "sector_mix": dict(secs),
               "note": "hand-built era book (pre-SEC-XBRL); era-documented "
                       "PE/g, computed betas, Adam wide-moat review, "
                       "no hindsight"}
        json.dump(out, open(os.path.join(
            HERE, f"books_growth_{year}.json"), "w"), indent=1)
        print(f"{year}: 16 stocks, anti-bubble {ab}/16, sectors {dict(secs)}")
    json.dump(BETAS, open(os.path.join(HERE, "deep_betas.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
