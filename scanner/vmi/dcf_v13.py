"""StockOracle DCF-20yr IV model, calibration v13 (2026-07).

Fitted on 36 large-cap tickers against StockOracle "Base IV" screenshots:
36/36 within +/-7 pct, 32/36 within +/-5 pct; the original 11 benchmarks all
inside +/-7 pct. Structure of the DCF itself is verified to the cent (Visa,
Lesson 5 p.6): base flow/share compounded from YEAR 1 at g for yrs 1-10,
then 4 pct yrs 11-20, NO terminal value, CAPM discount (Rf 3.608 pct +
beta x MRP 2.728 pct), IV/sh = PV/sh - (ST+LT debt)/sh + (cash+STI)/sh.

Model = (a) deterministic growth blend over analyst estimates + fundamentals
(net margin, capex/OCF, revenue 5y CAGR) with sector intercepts and
sector x margin / sector x capex interactions, combined with
(c) continuous per-sector base-flow mix over 10 components:
annual SEC flows (ocf/fcf/ni), finviz forward NI, stockanalysis.com TTM
flows (ocfT/fcfT/niT) and 3y SEC averages (ocf3/fcf3/ni3).
No invented caps or minimums; the only clamp (-12..80 on g) was part of the
fitted model specification during calibration.

Calibration code: scanner/calib/blend_fit_v13.py -> fit_v13.json.
"""
import math
from typing import Dict, List, Optional

_P = {
 "weights": [
  0.9893824705122276,
  3.5611196282207858,
  -0.27669630316380567,
  -4.920727872854131,
  -0.05077628013627941,
  -1.064232074467428,
  -0.030382639800254605,
  -0.1513766986726078,
  -0.5134835361062605,
  5.44625671389802,
  -0.18313199080125417,
  0.0,
  -0.1625292629883426,
  1.7924015426805655,
  1.0607980916466455,
  58.55731721310323,
  -7.344085775178269,
  -13.074391834591381,
  0.0,
  0.0,
  -1.7760846800423333,
  2.3416830951867493,
  10.102257521516103,
  0.0,
  0.8028282178104024,
  3.565124941539211,
  0.18568834526067168,
  0.0,
  -0.5565629012427111,
  -2.7333980034104792,
  -4.083985758023487,
  -8.440160387707142,
  -1.511529466319265,
  -0.5299114847381228,
  0.28066762719526156,
  0.0,
  0.4523899875814831,
  -0.04810485133591924,
  1.7435266482712939
 ],
 "feature_names": [
  "g5",
  "sqrt_g5",
  "eny",
  "sqrt_eny",
  "ety",
  "sqrt_ety",
  "ocfC",
  "fcfC",
  "marg",
  "sqrt_marg",
  "co",
  "sqrt_co",
  "rev5C",
  "sqrt_rev5C",
  "const",
  "sec_consumer",
  "sec_fin-net",
  "sec_health",
  "sec_industrial",
  "sec_retail",
  "sec_tech-hw",
  "sec_tech-net",
  "sec_tech-sw",
  "secXmarg_consumer",
  "secXmarg_fin-net",
  "secXmarg_health",
  "secXmarg_industrial",
  "secXmarg_retail",
  "secXmarg_tech-hw",
  "secXmarg_tech-net",
  "secXmarg_tech-sw",
  "secXco_consumer",
  "secXco_fin-net",
  "secXco_health",
  "secXco_industrial",
  "secXco_retail",
  "secXco_tech-hw",
  "secXco_tech-net",
  "secXco_tech-sw"
 ],
 "sectors": [
  "consumer",
  "fin-net",
  "health",
  "industrial",
  "retail",
  "tech-hw",
  "tech-net",
  "tech-sw"
 ],
 "components": [
  "ocf",
  "fcf",
  "ni",
  "fwdni",
  "ocfT",
  "fcfT",
  "niT",
  "ocf3",
  "fcf3",
  "ni3"
 ],
 "alphas": {
  "consumer": [
   0.9999999999999984,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0
  ],
  "fin-net": [
   0.0,
   0.0,
   0.2400748470900667,
   0.7599251529099335,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0
  ],
  "health": [
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   1.0,
   0.0,
   0.0
  ],
  "industrial": [
   0.9181055395425637,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.08189446045743631,
   0.0
  ],
  "retail": [
   0.9696809618109691,
   0.0,
   0.0,
   0.0,
   0.0,
   0.030319038189030922,
   0.0,
   0.0,
   0.0,
   0.0
  ],
  "tech-hw": [
   0.0,
   0.0,
   0.0,
   0.25022460517223055,
   0.0,
   0.0,
   0.0,
   0.6717867363064919,
   0.07798865852127762,
   0.0
  ],
  "tech-net": [
   0.0,
   0.0,
   0.0,
   -1.8312193426426855e-14,
   1.0000000000000182,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0
  ],
  "tech-sw": [
   0.0,
   0.0,
   0.0,
   0.0,
   1.0000000000000033,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0
  ]
 }
}

WEIGHTS: List[float] = _P["weights"]
FEATURE_NAMES: List[str] = _P["feature_names"]
SECTOR_GROUPS: List[str] = _P["sectors"]
COMPONENTS: List[str] = _P["components"]
ALPHAS: Dict[str, List[float]] = _P["alphas"]
_FALLBACK = {"ocfT": "ocf", "fcfT": "fcf", "niT": "ni",
             "ocf3": "ocf", "fcf3": "fcf", "ni3": "ni", "fwdni": "ni"}
RF, MRP = 3.608, 2.728
G_LO, G_HI = -12.0, 80.0   # part of the fitted spec (active in calibration)


def _sg(x: float) -> float:
    return math.copysign(math.sqrt(abs(x)), x)


def sector_group(sector: str, industry: str) -> str:
    """Map GICS sector/sub-industry (Wikipedia) or finviz free-text sector
    to the 8 calibration groups. Reproduces the hand labels of all 36
    calibration tickers exactly."""
    s = (sector or "").lower()
    i = (industry or "").lower()
    if "health" in s:
        # VEEV (Health Care Technology) behaves like software
        return "tech-sw" if "technology" in i else "health"
    if "information technology" in s or s == "technology":
        if any(k in i for k in ("semiconductor", "hardware",
                                "communications equipment", "electronic")):
            return "tech-hw"
        return "tech-sw"
    if "communication" in s:
        return "tech-net"
    if "financial" in s:
        return "fin-net"
    if "consumer discretionary" in s or "consumer cyclical" in s:
        if any(k in i for k in ("broadline retail", "internet", "hotels",
                                "travel", "casinos")):
            return "tech-net"          # AMZN, BKNG, MELI platform economics
        if "distributor" in i:
            return "industrial"        # POOL
        if "retail" in i:
            return "retail"            # AZO
        return "consumer"              # NKE etc.
    if "consumer staples" in s or "consumer defensive" in s:
        return "consumer"              # CELH; no separate staples group
    # Industrials, Materials, Energy, Utilities, Real Estate leftovers, etc.
    return "industrial"


def _cagr5(series_newest_first: Optional[List[Optional[float]]]) -> float:
    """5-interval CAGR (idx0 vs idx5) in percent — calibration definition."""
    s = series_newest_first or []
    if len(s) >= 6 and s[0] and s[5] and s[0] > 0 and s[5] > 0:
        return ((s[0] / s[5]) ** 0.2 - 1) * 100
    return 0.0


def _avg3(series_newest_first: Optional[List[Optional[float]]]) -> Optional[float]:
    vals = [v for v in (series_newest_first or [])[:3] if v is not None]
    return sum(vals) / len(vals) if len(vals) >= 2 else None


def compute_iv(*, sector: str, industry: str, shares: float,
               beta: Optional[float], g5: Optional[float],
               eny: Optional[float], ety: Optional[float],
               fwd_eps: Optional[float],
               ocf_series: List[Optional[float]],
               capex_series: List[Optional[float]],
               ni_series: List[Optional[float]],
               rev_series: List[Optional[float]],
               cash: float, sti: float, std: float, ltd: float,
               ttm: Optional[Dict[str, Optional[float]]] = None
               ) -> Optional[Dict[str, float]]:
    """Return dict(iv_ps, g_pct, disc_pct, base_desc, sector_group) or None.

    Series are newest-first annual values. `ttm` optionally carries
    {"ocf","capex","ni"} trailing-twelve-month values (capex negative,
    stockanalysis.com convention).
    """
    if not shares or shares <= 0:
        return None
    grp = sector_group(sector, industry)
    ocf0 = ocf_series[0] if ocf_series and ocf_series[0] is not None else None
    capex0 = (abs(capex_series[0])
              if capex_series and capex_series[0] is not None else 0.0)
    ni0 = ni_series[0] if ni_series and ni_series[0] is not None else None
    fcfs = [(o - abs(c)) if (o is not None and c is not None) else None
            for o, c in zip(ocf_series or [], capex_series or [])]
    flows: Dict[str, Optional[float]] = {
        "ocf": ocf0,
        "fcf": (ocf0 - capex0) if ocf0 is not None else None,
        "ni": ni0,
        "fwdni": (fwd_eps * shares) if fwd_eps else None,
        "ocf3": _avg3(ocf_series), "fcf3": _avg3(fcfs), "ni3": _avg3(ni_series),
    }
    if ttm and ttm.get("ocf") is not None:
        flows["ocfT"] = ttm["ocf"]
        cT = ttm.get("capex")
        flows["fcfT"] = ttm["ocf"] + cT if cT is not None else None
        flows["niT"] = ttm.get("ni")
    fps: Dict[str, float] = {}
    for c in COMPONENTS:
        v = flows.get(c)
        if v is None or v <= 0:
            v = flows.get(_FALLBACK.get(c, c))
        fps[c] = (v / shares) if (v is not None and v > 0) else 0.0

    alpha = ALPHAS[grp]
    base_ps = sum(a * fps[c] for a, c in zip(alpha, COMPONENTS))
    if base_ps <= 0:
        return None

    _g5 = g5 if g5 is not None else _cagr5(ocf_series)
    _eny = eny if eny is not None else 0.0
    _ety = ety if ety is not None else 0.0
    # One-time-item whipsaw guard: analyst "EPS this Y" massively up with
    # "EPS next Y" negative (e.g. AMZN 2026: +75% this / -16% next after the
    # Anthropic-stake gain) sits far outside the calibration envelope, where
    # eps_next_y was always >= +3.6% (inputs2.json domain). The concave
    # sqrt terms then explode g. Replace the distorted PAIR with their
    # 2-year compound annualized rate — pure arithmetic on the same two
    # analyst numbers, no new constants; identity when both years are normal.
    if _ety > 0 > _eny and (1 + _ety / 100) * (1 + _eny / 100) > 0:
        ann2 = (((1 + _ety / 100) * (1 + _eny / 100)) ** 0.5 - 1) * 100
        _eny = _ety = ann2
    ocfC = _cagr5(ocf_series)
    fcfC = _cagr5(fcfs)
    rev0 = rev_series[0] if rev_series and rev_series[0] is not None else None
    marg = (ni0 / rev0 * 100) if (rev0 and ni0 is not None) else 0.0
    co = (capex0 / ocf0 * 100) if (ocf0 and ocf0 > 0) else 0.0
    rev5C = _cagr5(rev_series)
    feats = [_g5, _sg(_g5), _eny, _sg(_eny), _ety, _sg(_ety), ocfC, fcfC,
             marg, _sg(marg), co, _sg(co), rev5C, _sg(rev5C), 1.0]
    dums = [1.0 if grp == s else 0.0 for s in SECTOR_GROUPS]
    x = feats + dums + [d * _sg(marg) for d in dums] + [d * _sg(co) for d in dums]
    g_pct = sum(w * f for w, f in zip(WEIGHTS, x))
    g_pct = max(G_LO, min(G_HI, g_pct))

    b = beta if beta is not None else 1.0
    disc = (RF + b * MRP) / 100.0
    g1 = g_pct / 100.0
    pv, f = 0.0, base_ps
    for yr in range(1, 21):
        f *= 1 + (g1 if yr <= 10 else 0.04)
        pv += f / (1 + disc) ** yr
    iv_ps = pv - ((std or 0) + (ltd or 0)) / shares + ((cash or 0) + (sti or 0)) / shares
    mix = {c: round(a, 3) for a, c in zip(alpha, COMPONENTS) if a > 0.01}
    base_desc = "+".join(f"{c}:{w}" for c, w in mix.items())
    return {"iv_ps": iv_ps, "g_pct": g_pct, "disc_pct": disc * 100,
            "base_desc": base_desc, "sector_group": grp}


# Lowest discount rate observed in Adam's beta lookup table (VMI Master
# Document, Lesson 5 §4.7: beta 0.28 -> 5.8%). Pure CAPM with our RF/MRP
# gives 4.37% at that beta, so the table clearly floors low-beta names.
# The other observed rows (0.83->6.1, 1.1->6.6, 1.4->7.4, 1.49->7.7)
# match RF + beta*MRP within 0.25pp, so a floor is the only correction.
DISC_FLOOR = 5.8


def compute_iv_direct(*, shares: float, beta: Optional[float],
                      g5: Optional[float],
                      ni_series: List[Optional[float]],
                      cash: float, sti: float, std: float, ltd: float,
                      ttm_ni: Optional[float] = None,
                      ocf_series: Optional[List[Optional[float]]] = None,
                      capex_series: Optional[List[Optional[float]]] = None,
                      ttm_ocf: Optional[float] = None
                      ) -> Optional[Dict[str, float]]:
    """DIRECT DCF per Adam Khoo's Lesson 5 procedure (VMI Master Document):
    no fitted blend, no sector terms — the taught manual method verbatim.

    Base flow (Adam's decision sequence, Lesson 5 §11):
      1. DEFAULT: normalized Free Cash Flow = latest 12-month cash flow
         from operations - 5-YEAR AVERAGE capex (§5.2 — automatically
         equals plain FCF when capex is even; fixes lumpy-capex names
         like Amazon where latest-year FCF is artificially crushed).
      2. FALLBACK: net income (Discounted Net Income, §7) when operating
         cash flow is unavailable, non-positive, or LESS CONSISTENT than
         net income (§7.1: "use whichever series is the more consistent",
         measured as the fraction of up-years in each series).

    Growth bands (§4.6): years 1-5 at g (average of GuruFocus + Finviz +
    Zacks projected 3-5y growth, see growth.py); years 6-10 same g capped
    at 15%; years 11-20 at 4% (nominal GDP + 2%). No terminal value —
    Adam values 20 years only (§4.8).

    Discount rate (§4.7): risk-free 3.608% + beta x MRP 2.728%, floored
    at 5.8% (the lowest row in Adam's published beta table). Net cash
    added, total debt subtracted, divided by shares outstanding.

    Accuracy note: with StockOracle's own growth numbers this recipe has
    ~24% median abs error vs their Base IV (the refined calibrated blend
    is 36/36 within +/-7%). It is intentionally uncorrected — it shows
    what the raw published inputs say through the taught formula.
    """
    if not shares or shares <= 0 or g5 is None:
        return None

    def _pos(vals):
        return [v for v in (vals or []) if v is not None]

    def _upfrac(vals):
        """Fraction of positive YoY changes, series given newest-first."""
        s = list(reversed(_pos(vals)))
        if len(s) < 3:
            return None
        ups = sum(1 for a, b2 in zip(s, s[1:]) if b2 > a)
        return ups / (len(s) - 1)

    # --- Base flow: normalized FCF default, NI fallback (Adam §11) -----
    ocf_vals = _pos(ocf_series)
    ni_vals = _pos(ni_series)
    ocf_latest = (ttm_ocf if ttm_ocf is not None and ttm_ocf > 0
                  else (ocf_vals[0] if ocf_vals and ocf_vals[0] > 0
                        else None))
    capex_vals = [abs(v) for v in _pos(capex_series)][:5]
    capex_avg = sum(capex_vals) / len(capex_vals) if capex_vals else 0.0

    base = None
    base_desc = None
    ocf_cons = _upfrac(ocf_series)
    ni_cons = _upfrac(ni_series)
    use_ni = (ocf_latest is None or
              (ocf_cons is not None and ni_cons is not None
               and ni_cons > ocf_cons))
    if not use_ni and ocf_latest is not None:
        nfcf = ocf_latest - capex_avg
        if nfcf > 0:
            base, base_desc = nfcf, "normalized FCF (CFO - 5y avg capex)"
    if base is None:
        ni_latest = (ttm_ni if ttm_ni is not None and ttm_ni > 0
                     else (ni_vals[0] if ni_vals and ni_vals[0] > 0
                           else None))
        if ni_latest is not None:
            base, base_desc = ni_latest, "net income (DNI)"
    if base is None:
        return None

    base_ps = base / shares
    b = beta if beta is not None else 1.0
    disc = max(RF + b * MRP, DISC_FLOOR) / 100.0
    g1 = g5 / 100.0             # years 1-5: full projected growth
    g2 = min(g5, 15.0) / 100.0  # years 6-10: same rate capped at 15%
    pv, f = 0.0, base_ps
    for yr in range(1, 21):
        g_yr = g1 if yr <= 5 else (g2 if yr <= 10 else 0.04)
        f *= 1 + g_yr
        pv += f / (1 + disc) ** yr
    iv_ps = pv - ((std or 0) + (ltd or 0)) / shares \
        + ((cash or 0) + (sti or 0)) / shares
    return {"iv_ps": iv_ps, "g_pct": g5, "disc_pct": disc * 100,
            "base_desc": base_desc}
