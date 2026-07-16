"""Two-stage growth blend fit, StockOracle-style.

Structure (verified to the cent on the Visa screenshot):
  IV = sum_{y=1..20} flow*(prod growth)/ (1+disc)^y  - debt/sh + cash&STI/sh
  g yr1-5 = blend1(features), g yr6-10 = blend2(features), yr11-20 = 4%
  disc = CAPM = 3.608% + beta * 2.728%

Features are all real public data (finviz analyst estimates, Yahoo
fundamentals, historical CAGRs). Weights shared across ALL tickers - the
blend is the model, no per-ticker edits.
"""
import json, os
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(__file__)
Y = json.load(open(os.path.join(HERE, "yahoo_flows.json")))
C = json.load(open(os.path.join(HERE, "calib_inputs.json")))
RF, MRP = 0.03608, 0.02728

FEATS = ["g5", "ety", "eny", "revg", "eg", "ocf_cagr", "fcf_cagr"]

def cagr(seq):
    seq = [x for x in (seq or []) if x is not None]
    if len(seq) >= 3 and seq[-1] > 0 and seq[0] > 0:
        return ((seq[0] / seq[-1]) ** (1 / (len(seq) - 1)) - 1) * 100
    return None

D = {}
for t, y in Y.items():
    c = C[t]
    sh = y["shares"]; disc = RF + (y["beta"] or 1.0) * MRP
    ocf = y.get("ocf_ttm_q") or y["ocf_ttm"]
    fcf = y.get("fcf_ttm_q") or y["fcf_ttm_yahoo"]
    caps = [abs(x) for x in (y.get("capex_hist") or []) if x is not None]
    fcfavg = ocf - sum(caps) / len(caps) if caps else None
    ni = y.get("ni_ttm")
    feats = {
        "g5": c["g5"], "ety": c.get("eps_this_y") or 0.0, "eny": c.get("eps_next_y") or 0.0,
        "revg": (y.get("revenueGrowth") or 0) * 100, "eg": (y.get("earningsGrowth") or 0) * 100,
        "ocf_cagr": cagr(y.get("ocf_hist")) or 0.0, "fcf_cagr": cagr(y.get("fcf_hist")) or 0.0,
    }
    D[t] = dict(sh=sh, disc=disc, debt_ps=y["totalDebt"] / sh, cash_ps=y["totalCash"] / sh,
                bases=dict(ocf=ocf, fcf=fcf, fcfavg=fcfavg, ni=ni),
                x=np.array([feats[f] for f in FEATS]), tgt=c["target_iv"], feats=feats)

def iv(base_ps, g1, g2, disc, debt_ps, cash_ps):
    pv, f = 0.0, base_ps
    for yr in range(1, 21):
        f *= 1 + (g1 if yr <= 5 else g2 if yr <= 10 else 0.04)
        pv += f / (1 + disc) ** yr
    return pv - debt_ps + cash_ps

def eval_params(p, base_rule):
    w1, b1 = p[:len(FEATS)], p[len(FEATS)]
    w2, b2 = p[len(FEATS)+1:2*len(FEATS)+1], p[-1]
    errs = {}
    for t, d in D.items():
        g1 = (float(w1 @ d["x"]) + b1) / 100.0
        g2 = (float(w2 @ d["x"]) + b2) / 100.0
        bval = base_rule(d)
        v = iv(bval / d["sh"], g1, g2, d["disc"], d["debt_ps"], d["cash_ps"])
        errs[t] = (v - d["tgt"]) / d["tgt"] * 100
    return errs

def objective(p, base_rule, lam=0.0005):
    errs = eval_params(p, base_rule)
    e = np.array(list(errs.values()))
    return float(np.sqrt((e ** 2).mean()) + 0.5 * np.abs(e).max() + lam * np.sum(np.array(p) ** 2))

RULES = {
    "fcf_first": lambda d: (d["bases"]["fcf"] if d["bases"]["fcf"] and d["bases"]["fcf"] > 0
                            else d["bases"]["fcfavg"] if d["bases"]["fcfavg"] and d["bases"]["fcfavg"] > 0
                            else d["bases"]["ni"]),
    "ocf_always": lambda d: d["bases"]["ocf"],
    "fcfavg_first": lambda d: (d["bases"]["fcfavg"] if d["bases"]["fcfavg"] and d["bases"]["fcfavg"] > 0
                               else d["bases"]["ni"]),
}

n = 2 * len(FEATS) + 2
results = {}
for rname, rule in RULES.items():
    best = None
    rng = np.random.default_rng(42)
    for trial in range(30):
        p0 = rng.normal(0, 0.3, n)
        p0[len(FEATS)] = 5.0; p0[-1] = 5.0  # intercepts
        r = minimize(objective, p0, args=(rule,), method="Nelder-Mead",
                     options=dict(maxiter=20000, xatol=1e-6, fatol=1e-8))
        errs = eval_params(r.x, rule)
        mx = max(abs(e) for e in errs.values())
        if best is None or mx < best[0]:
            best = (mx, r.x, errs)
    results[rname] = best
    print(f"== rule {rname}: max|err| = {best[0]:.2f}%")
    for t, e in best[2].items():
        print(f"   {t:6} {e:+6.2f}%")

# save best overall
rname = min(results, key=lambda k: results[k][0])
mx, p, errs = results[rname]
out = dict(rule=rname, feats=FEATS,
           w1=list(p[:len(FEATS)]), b1=float(p[len(FEATS)]),
           w2=list(p[len(FEATS)+1:2*len(FEATS)+1]), b2=float(p[-1]),
           max_abs_err=mx, errs=errs)
json.dump(out, open(os.path.join(HERE, "blend_params.json"), "w"), indent=1)
print("\nBEST:", rname, "max|err|", round(mx, 2), "-> saved blend_params.json")
