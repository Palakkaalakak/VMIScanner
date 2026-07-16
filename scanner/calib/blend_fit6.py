"""Blend fit v6 - ridge-regularized v5 with sanity constraints.

Same deterministic structure as v5 (base = OCF * m(fundamental ratios),
g = blend(growth features)) but both regressions are ridge-regularized.
Sweep lambda; keep the MOST regularized (tamest, best-generalizing) model
whose max benchmark error is <= 3.5%. Also report predicted g / m sanity.
"""
import json, os
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

G_FEATS = ["g5", "sqrt_g5", "eny", "sqrt_eny", "ety", "sqrt_ety", "revg", "ocfC", "fcfC", "one"]
B_FEATS = ["one", "capex_ocf", "ni_ocf", "fcf_ocf", "avgcapex_ocf"]

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
    xg = np.array([g5, np.sqrt(max(g5,0)), eny, np.sqrt(abs(eny))*np.sign(eny),
                   ety, np.sqrt(abs(ety))*np.sign(ety), (y.get("revenueGrowth") or 0)*100,
                   cagr(y.get("ocf_hist")) or 0.0, cagr(y.get("fcf_hist")) or 0.0, 1.0])
    xb = np.array([1.0, capex/ocf, ni/ocf, fcf/ocf, avgcap/ocf])
    D[t] = dict(sh=sh, disc=disc, debt_ps=y["totalDebt"]/sh, cash_ps=y["totalCash"]/sh,
                ocf_ps=ocf/sh, xg=xg, xb=xb, tgt=c["target_iv"])

tickers = list(D)
Xg = np.array([D[t]["xg"] for t in tickers])
Xb = np.array([D[t]["xb"] for t in tickers])

def ridge(X, yv, lam):
    n = X.shape[1]
    P = np.eye(n); P[-1, -1] = 0.0  # don't penalize intercept ("one" is last for Xg, first for Xb)
    return np.linalg.solve(X.T @ X + lam * P, X.T @ yv)

def ridge_b(X, yv, lam):
    n = X.shape[1]
    P = np.eye(n); P[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + lam * P, X.T @ yv)

def run(lam_g, lam_b, iters=60):
    m = np.ones(len(tickers)) * 0.85
    wg = wb = None
    for it in range(iters):
        gs = []
        for i, t in enumerate(tickers):
            d = D[t]
            base = d["ocf_ps"] * float(np.clip(m[i], 0.05, 3.0))
            gs.append(brentq(lambda g: iv(base, g, d["disc"], d["debt_ps"], d["cash_ps"]) - d["tgt"], -0.6, 2.5) * 100)
        wg = ridge(Xg, np.array(gs), lam_g)
        ghat = Xg @ wg
        ms = []
        for i, t in enumerate(tickers):
            d = D[t]; g = ghat[i] / 100.0
            try:
                b = brentq(lambda b_: iv(b_, g, d["disc"], d["debt_ps"], d["cash_ps"]) - d["tgt"],
                           1e-6, d["ocf_ps"] * 6)
                ms.append(b / d["ocf_ps"])
            except Exception:
                ms.append(m[i])
        wb = ridge_b(Xb, np.array(ms), lam_b)
        m_new = Xb @ wb
        if np.max(np.abs(m_new - m)) < 1e-12: m = m_new; break
        m = m_new
    errs = {}
    gvals, mvals = {}, {}
    for i, t in enumerate(tickers):
        d = D[t]
        g = float(d["xg"] @ wg) / 100.0
        mult = float(d["xb"] @ wb)
        base = d["ocf_ps"] * mult
        v = iv(base, g, d["disc"], d["debt_ps"], d["cash_ps"]) if base > 0 else float("nan")
        errs[t] = (v - d["tgt"]) / d["tgt"] * 100
        gvals[t] = g * 100; mvals[t] = mult
    return wg, wb, errs, gvals, mvals

best = None
for lam_g in (300, 100, 30, 10, 3, 1, 0.3, 0.1):
    for lam_b in (3, 1, 0.3, 0.1, 0.03, 0.01):
        try:
            wg, wb, errs, gvals, mvals = run(lam_g, lam_b)
        except Exception:
            continue
        mx = max(abs(e) for e in errs.values())
        wnorm = float(np.sum(wg[:-1]**2)) + float(np.sum(wb[1:]**2))
        if mx <= 3.5:
            score = wnorm  # prefer tamest weights
            if best is None or score < best[0]:
                best = (score, lam_g, lam_b, wg, wb, errs, gvals, mvals, mx)

if best is None:
    print("no config met 3.5% - rerun with looser bound")
else:
    score, lam_g, lam_b, wg, wb, errs, gvals, mvals, mx = best
    print(f"chosen lam_g={lam_g} lam_b={lam_b}  max|err|={mx:.2f}%")
    print("g-weights:", {n: round(float(v),4) for n,v in zip(G_FEATS, wg)})
    print("b-weights:", {n: round(float(v),4) for n,v in zip(B_FEATS, wb)})
    for t in tickers:
        print(f"  {t:6} g={gvals[t]:6.2f}%  m={mvals[t]:5.2f}  err={errs[t]:+6.2f}%")
    json.dump(dict(g_feats=G_FEATS, b_feats=B_FEATS, wg=list(map(float,wg)),
                   wb=list(map(float,wb)), lam_g=lam_g, lam_b=lam_b,
                   errs=errs, max_abs_err=mx),
              open(os.path.join(HERE, "blend_params_v6.json"), "w"), indent=1)
    print("saved blend_params_v6.json")
