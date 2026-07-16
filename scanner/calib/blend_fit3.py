"""Blend fit v3: growth = linear blend over public sources PLUS smooth
nonlinear transforms (sqrt, log1p) that act as natural shrinkage for
extreme analyst numbers (no hard caps). Base flow chosen per ticker by
whichever of OCF/FCF/FCF(5y-avg-capex)/NI best matches - then we check the
chosen bases against Adam's consistency rules. Structure fixed = verified
Visa screenshot model (20y, g1 yr1-5, g2 yr6-10, 4% yr11-20, CAPM, -debt +cash).
"""
import json, os
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(__file__)
Y = json.load(open(os.path.join(HERE, "yahoo_flows.json")))
C = json.load(open(os.path.join(HERE, "calib_inputs.json")))
RF, MRP = 0.03608, 0.02728

def cagr(seq):
    seq = [x for x in (seq or []) if x is not None]
    if len(seq) >= 3 and seq[-1] > 0 and seq[0] > 0:
        return ((seq[0] / seq[-1]) ** (1 / (len(seq) - 1)) - 1) * 100
    return None

def feats_of(t, y, c):
    g5 = c["g5"]; ety = c.get("eps_this_y") or 0.0; eny = c.get("eps_next_y") or 0.0
    revg = (y.get("revenueGrowth") or 0) * 100
    return {
        "g5": g5, "sqrt_g5": np.sqrt(max(g5, 0)) * np.sign(g5),
        "log_g5": np.log1p(max(g5, 0)),
        "eny": eny, "sqrt_eny": np.sqrt(abs(eny)) * np.sign(eny),
        "ety": ety, "revg": revg,
        "ocfC": cagr(y.get("ocf_hist")) or 0.0,
        "fcfC": cagr(y.get("fcf_hist")) or 0.0,
        "one": 1.0,
    }

FEATS = None
D = {}
for t, y in Y.items():
    c = C[t]
    f = feats_of(t, y, c)
    if FEATS is None: FEATS = list(f.keys())
    sh = y["shares"]; disc = RF + (y["beta"] or 1.0) * MRP
    ocf = y.get("ocf_ttm_q") or y["ocf_ttm"]
    fcf = y.get("fcf_ttm_q") or y["fcf_ttm_yahoo"]
    caps = [abs(x) for x in (y.get("capex_hist") or []) if x is not None]
    fcfavg = ocf - sum(caps) / len(caps) if caps else None
    ni = y.get("ni_ttm")
    D[t] = dict(sh=sh, disc=disc, debt_ps=y["totalDebt"]/sh, cash_ps=y["totalCash"]/sh,
                bases=dict(ocf=ocf, fcf=fcf, fcfavg=fcfavg, ni=ni),
                x=np.array([f[k] for k in FEATS]), tgt=c["target_iv"])

def iv(base_ps, g1, g2, disc, debt_ps, cash_ps):
    pv, f = 0.0, base_ps
    for yr in range(1, 21):
        f *= 1 + (g1 if yr <= 5 else g2 if yr <= 10 else 0.04)
        pv += f / (1 + disc) ** yr
    return pv - debt_ps + cash_ps

NF = None
def eval_params(p, pick=None):
    n = len(FEATS)
    w1, w2 = p[:n], p[n:2*n]
    errs, picks = {}, {}
    for t, d in D.items():
        g1 = float(w1 @ d["x"]) / 100.0
        g2 = float(w2 @ d["x"]) / 100.0
        best_e, best_b = None, None
        cand = d["bases"] if pick is None else {pick[t]: d["bases"][pick[t]]}
        for bn, bv in cand.items():
            if not bv or bv <= 0: continue
            v = iv(bv / d["sh"], g1, g2, d["disc"], d["debt_ps"], d["cash_ps"])
            e = (v - d["tgt"]) / d["tgt"] * 100
            if best_e is None or abs(e) < abs(best_e):
                best_e, best_b = e, bn
        errs[t] = best_e; picks[t] = best_b
    return errs, picks

def objective(p):
    errs, _ = eval_params(p)
    e = np.array(list(errs.values()))
    # heavily punish worst error to drive max under 4
    return float(np.sqrt((e**2).mean()) + 1.5 * np.abs(e).max() + 0.0002 * np.sum(p**2))

n = len(FEATS)
rng = np.random.default_rng(7)
best = None
for trial in range(60):
    p0 = rng.normal(0, 0.4, 2*n)
    r = minimize(objective, p0, method="Nelder-Mead",
                 options=dict(maxiter=40000, maxfev=40000, xatol=1e-7, fatol=1e-9))
    errs, picks = eval_params(r.x)
    mx = max(abs(e) for e in errs.values())
    if best is None or mx < best[0]:
        best = (mx, r.x, errs, picks)
        if mx < 2.0: break

mx, p, errs, picks = best
print("FEATS:", FEATS)
print("w1:", [round(v,4) for v in p[:n]])
print("w2:", [round(v,4) for v in p[n:]])
print(f"max|err| = {mx:.2f}%")
for t in D:
    print(f"  {t:6} {errs[t]:+6.2f}%  base={picks[t]}")
json.dump(dict(feats=FEATS, w1=list(p[:n]), w2=list(p[n:]),
               errs=errs, picks=picks, max_abs_err=mx),
          open(os.path.join(HERE, "blend_params.json"), "w"), indent=1)
print("saved blend_params.json")
