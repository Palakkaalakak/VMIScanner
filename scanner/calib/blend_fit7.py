"""Blend fit v7 - subset search for a TAME deterministic model.

Try all small subsets of growth features (plus intercept) and base features
(plus intercept), fit by alternating exact least squares (same as v5), and
keep models with max|err| <= 4.0 ranked by (a) sanity of predicted growth
across benchmarks (no negatives, none > 35%), (b) smallest weight norm.
"""
import itertools, json, os
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(__file__)
Y = json.load(open(os.path.join(HERE, "yahoo_flows.json")))
C = json.load(open(os.path.join(HERE, "calib_inputs.json")))
RF, MRP = 0.03608, 0.02728

def cagr(seq):
    seq = [x for x in (seq or []) if x is not None]
    if len(seq) >= 3 and seq[-1] > 0 and seq[0] > 0:
        return ((seq[0]/seq[-1])**(1/(len(seq)-1))-1)*100
    return None

def iv(base_ps, g, disc, debt_ps, cash_ps):
    pv, f = 0.0, base_ps
    for yr in range(1, 21):
        f *= 1 + (g if yr <= 10 else 0.04)
        pv += f / (1 + disc) ** yr
    return pv - debt_ps + cash_ps

ALL_G = ["g5", "sqrt_g5", "eny", "sqrt_eny", "ety", "sqrt_ety", "revg", "ocfC", "fcfC"]
ALL_B = ["capex_ocf", "ni_ocf", "fcf_ocf", "avgcapex_ocf"]

D = {}
for t, y in Y.items():
    c = C[t]
    sh = y["shares"]; disc = RF + (y["beta"] or 1.0) * MRP
    ocf = y.get("ocf_ttm_q") or y["ocf_ttm"]
    fcf = y.get("fcf_ttm_q") or y["fcf_ttm_yahoo"] or 0.0
    capex = abs(y.get("capex_ttm_q") or (ocf - fcf if fcf else 0.0))
    caps = [abs(x) for x in (y.get("capex_hist") or []) if x is not None]
    avgcap = sum(caps)/len(caps) if caps else capex
    ni = y.get("ni_ttm") or 0.0
    g5 = c["g5"]; ety = c.get("eps_this_y") or 0.0; eny = c.get("eps_next_y") or 0.0
    gf = {"g5": g5, "sqrt_g5": np.sqrt(max(g5,0)), "eny": eny,
          "sqrt_eny": np.sqrt(abs(eny))*np.sign(eny), "ety": ety,
          "sqrt_ety": np.sqrt(abs(ety))*np.sign(ety),
          "revg": (y.get("revenueGrowth") or 0)*100,
          "ocfC": cagr(y.get("ocf_hist")) or 0.0, "fcfC": cagr(y.get("fcf_hist")) or 0.0}
    bf = {"capex_ocf": capex/ocf, "ni_ocf": ni/ocf, "fcf_ocf": fcf/ocf, "avgcapex_ocf": avgcap/ocf}
    D[t] = dict(sh=sh, disc=disc, debt_ps=y["totalDebt"]/sh, cash_ps=y["totalCash"]/sh,
                ocf_ps=ocf/sh, gf=gf, bf=bf, tgt=c["target_iv"])

tickers = list(D)

def run(gsub, bsub, iters=50):
    Xg = np.array([[D[t]["gf"][f] for f in gsub] + [1.0] for t in tickers])
    Xb = np.array([[1.0] + [D[t]["bf"][f] for f in bsub] for t in tickers])
    m = np.ones(len(tickers)) * 0.85
    for it in range(iters):
        gs = []
        for i, t in enumerate(tickers):
            d = D[t]
            base = d["ocf_ps"] * float(np.clip(m[i], 0.05, 3.0))
            gs.append(brentq(lambda g: iv(base, g, d["disc"], d["debt_ps"], d["cash_ps"]) - d["tgt"], -0.6, 2.5) * 100)
        wg, *_ = np.linalg.lstsq(Xg, np.array(gs), rcond=None)
        ghat = Xg @ wg
        ms = []
        for i, t in enumerate(tickers):
            d = D[t]; g = ghat[i] / 100.0
            try:
                b = brentq(lambda b_: iv(b_, g, d["disc"], d["debt_ps"], d["cash_ps"]) - d["tgt"],
                           1e-6, d["ocf_ps"] * 6)
                ms.append(b / d["ocf_ps"])
            except Exception:
                ms.append(float(np.clip(m[i], 0.05, 3.0)))
        wb, *_ = np.linalg.lstsq(Xb, np.array(ms), rcond=None)
        m_new = Xb @ wb
        if np.max(np.abs(m_new - m)) < 1e-12: m = m_new; break
        m = m_new
    errs, gvals, mvals = {}, {}, {}
    for i, t in enumerate(tickers):
        d = D[t]
        g = float(Xg[i] @ wg) / 100.0
        mult = float(Xb[i] @ wb)
        base = d["ocf_ps"] * mult
        if base <= 0: return None
        v = iv(base, g, d["disc"], d["debt_ps"], d["cash_ps"])
        errs[t] = (v - d["tgt"]) / d["tgt"] * 100
        gvals[t] = g*100; mvals[t] = mult
    return wg, wb, errs, gvals, mvals

results = []
gsubs = [list(s) for k in (2,3,4) for s in itertools.combinations(ALL_G, k)]
bsubs = [list(s) for k in (1,2) for s in itertools.combinations(ALL_B, k)]
print(f"searching {len(gsubs)}x{len(bsubs)} combos...")
for gsub in gsubs:
    for bsub in bsubs:
        try:
            r = run(gsub, bsub)
        except Exception:
            continue
        if r is None: continue
        wg, wb, errs, gvals, mvals = r
        mx = max(abs(e) for e in errs.values())
        if mx > 4.0: continue
        gv = np.array(list(gvals.values())); mv = np.array(list(mvals.values()))
        sane = (gv.min() > 0) and (gv.max() < 35) and (mv.min() > 0.2) and (mv.max() < 1.6)
        wnorm = float(np.sum(wg[:-1]**2) + np.sum(wb[1:]**2))
        results.append(dict(gsub=gsub, bsub=bsub, mx=mx, sane=sane, wnorm=wnorm,
                            wg=list(map(float,wg)), wb=list(map(float,wb)),
                            errs=errs, gvals=gvals, mvals=mvals))
results.sort(key=lambda r: (not r["sane"], r["wnorm"]))
print(f"{len(results)} combos meet <=4%; showing top 5")
for r in results[:5]:
    print(f"\n g={r['gsub']} b={r['bsub']} max|err|={r['mx']:.2f}% sane={r['sane']} wnorm={r['wnorm']:.1f}")
    print("  wg:", [round(v,3) for v in r["wg"]], " wb:", [round(v,3) for v in r["wb"]])
    for t in tickers:
        print(f"   {t:6} g={r['gvals'][t]:6.2f}% m={r['mvals'][t]:5.2f} err={r['errs'][t]:+6.2f}%")
if results:
    json.dump(results[0], open(os.path.join(HERE, "blend_params_v7.json"), "w"), indent=1)
    print("\nsaved best -> blend_params_v7.json")
