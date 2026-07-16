"""Fit a growth-rate blend (shared weights over public sources) plus a
per-ticker base-flow choice (OCF / FCF / OCF-5yavgCapex / NI, per Adam's
method rules) to reproduce StockOracle Base IVs. Verified DCF20 structure:
g yr1-10 (blend), 4% yr11-20, CAPM disc, -debt/sh +cash&STI/sh, 20y, no TV.
"""
import itertools, json, os
HERE = os.path.dirname(__file__)
Y = json.load(open(os.path.join(HERE, "yahoo_flows.json")))
C = json.load(open(os.path.join(HERE, "calib_inputs.json")))
RF, MRP = 0.03608, 0.02728

def iv(base, g, disc, debt_ps, cash_ps):
    pv, f = 0.0, base
    for yr in range(1, 21):
        f *= (1 + (g if yr <= 10 else 0.04))
        pv += f / (1 + disc) ** yr
    return pv - debt_ps + cash_ps

# assemble per-ticker data
D = {}
for t, y in Y.items():
    c = C[t]
    sh = y["shares"]; disc = RF + (y["beta"] or 1.0) * MRP
    ocf = y.get("ocf_ttm_q") or y["ocf_ttm"]
    fcf = y.get("fcf_ttm_q") or y["fcf_ttm_yahoo"]
    caps = [abs(x) for x in (y.get("capex_hist") or []) if x is not None]
    fcf_avg = ocf - sum(caps) / len(caps) if caps else None
    ni = y.get("ni_ttm")
    feats = {
        "g5": c["g5"],                      # finviz EPS next 5Y
        "ety": c.get("eps_this_y"),         # finviz EPS growth this year
        "eny": c.get("eps_next_y"),         # finviz EPS growth next year
        "revg": (y.get("revenueGrowth") or 0) * 100,
    }
    D[t] = dict(sh=sh, disc=disc, debt_ps=y["totalDebt"]/sh, cash_ps=y["totalCash"]/sh,
                bases={"ocf": ocf, "fcf": fcf, "fcfavg": fcf_avg, "ni": ni},
                feats=feats, tgt=c["target_iv"])

FEATS = ["g5", "ety", "eny", "revg"]
steps = [i / 10 for i in range(0, 11)]
best = None
for w in itertools.product(steps, repeat=len(FEATS)):
    if abs(sum(w) - 1.0) > 1e-9:
        continue
    errs, picks = {}, {}
    for t, d in D.items():
        gpct = sum(wi * (d["feats"][f] or 0) for wi, f in zip(w, FEATS))
        g = gpct / 100.0
        best_e, best_b = None, None
        for bname, bval in d["bases"].items():
            if not bval or bval <= 0:
                continue
            v = iv(bval / d["sh"], g, d["disc"], d["debt_ps"], d["cash_ps"])
            e = (v - d["tgt"]) / d["tgt"] * 100
            if best_e is None or abs(e) < abs(best_e):
                best_e, best_b = e, bname
        errs[t] = best_e; picks[t] = best_b
    mx = max(abs(e) for e in errs.values())
    if best is None or mx < best[0]:
        best = (mx, w, dict(errs), dict(picks))

mx, w, errs, picks = best
print("best weights:", dict(zip(FEATS, w)), "max|err| =", round(mx, 2))
for t in D:
    print(f"{t:6} err {errs[t]:+7.2f}%  base={picks[t]}")
