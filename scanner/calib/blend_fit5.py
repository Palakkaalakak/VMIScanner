"""Blend fit v5 - fully deterministic at scan time.

base_ps = OCF/sh * m,  m = wb . [1, capex/ocf, ni/ocf, fcf/ocf] (shared)
g yr1-10 = wg . growth features (shared), 4% yr11-20, CAPM disc,
-debt/sh +cash&STI/sh (verified Visa-screenshot DCF20 structure).

Alternating refinement:
  A) given base multiplier -> solve implied g per ticker -> least-squares wg
  B) given wg -> solve implied base per ticker -> least-squares wb over ratios
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

# init m from v4 picks if available
try:
    P4 = json.load(open(os.path.join(HERE, "blend_params.json")))
    picks = P4["picks"]
    m = []
    for t in tickers:
        d = D[t]; xb = d["xb"]
        mult = {"ocf": 1.0, "fcf": xb[3], "fcfavg": 1.0 - xb[4], "ni": xb[2]}[picks[t]]
        m.append(mult)
    m = np.array(m)
except Exception:
    m = np.ones(len(tickers)) * 0.85

wg = None; wb = None
for it in range(40):
    # A) implied g per ticker given base multiplier
    gs = []
    for i, t in enumerate(tickers):
        d = D[t]
        base = d["ocf_ps"] * max(m[i], 0.05)
        gs.append(brentq(lambda g: iv(base, g, d["disc"], d["debt_ps"], d["cash_ps"]) - d["tgt"], -0.6, 2.5) * 100)
    gs = np.array(gs)
    wg, *_ = np.linalg.lstsq(Xg, gs, rcond=None)
    ghat = Xg @ wg
    # B) implied base multiplier per ticker given predicted g
    ms = []
    for i, t in enumerate(tickers):
        d = D[t]; g = ghat[i] / 100.0
        try:
            b = brentq(lambda b_: iv(b_, g, d["disc"], d["debt_ps"], d["cash_ps"]) - d["tgt"],
                       1e-6, d["ocf_ps"] * 5)
            ms.append(b / d["ocf_ps"])
        except Exception:
            ms.append(m[i])
    ms = np.array(ms)
    wb, *_ = np.linalg.lstsq(Xb, ms, rcond=None)
    m_new = Xb @ wb
    if np.max(np.abs(m_new - m)) < 1e-10:
        m = m_new; break
    m = m_new

errs = {}
for i, t in enumerate(tickers):
    d = D[t]
    g = float(d["xg"] @ wg) / 100.0
    mult = float(d["xb"] @ wb)
    base = d["ocf_ps"] * mult
    v = iv(base, g, d["disc"], d["debt_ps"], d["cash_ps"]) if base > 0 else float("nan")
    errs[t] = (v - d["tgt"]) / d["tgt"] * 100
    print(f"  {t:6} g={g*100:6.2f}%  m={mult:5.2f}  err={errs[t]:+6.2f}%")
mx = max(abs(e) for e in errs.values())
print("g-weights:", {n: round(float(v),4) for n,v in zip(G_FEATS, wg)})
print("b-weights:", {n: round(float(v),4) for n,v in zip(B_FEATS, wb)})
print(f"max|err| = {mx:.2f}%  MAPE = {np.mean([abs(e) for e in errs.values()]):.2f}%")
json.dump(dict(g_feats=G_FEATS, b_feats=B_FEATS, wg=list(map(float,wg)),
               wb=list(map(float,wb)), errs=errs, max_abs_err=mx),
          open(os.path.join(HERE, "blend_params_v5.json"), "w"), indent=1)
print("saved blend_params_v5.json")
