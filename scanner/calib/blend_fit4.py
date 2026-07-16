"""Blend fit v4 (EM-style, exact linear algebra).

1. For each ticker and each candidate base flow (OCF / FCF / OCF-5yavgCapex
   / NI) solve the implied uniform growth g (yr1-10) that reproduces the
   StockOracle Base IV under the verified DCF20 structure.
2. Regress implied g on public growth features (least squares).
3. Reassign each ticker's base to whichever candidate's implied g is best
   predicted by the regression; repeat until stable.
Result: shared blend weights + a deterministic base-selection, no caps.
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
        return ((seq[0] / seq[-1]) ** (1 / (len(seq) - 1)) - 1) * 100
    return None

def iv(base_ps, g, disc, debt_ps, cash_ps):
    pv, f = 0.0, base_ps
    for yr in range(1, 21):
        f *= 1 + (g if yr <= 10 else 0.04)
        pv += f / (1 + disc) ** yr
    return pv - debt_ps + cash_ps

FEAT_NAMES = ["g5", "sqrt_g5", "eny", "sqrt_eny", "ety", "sqrt_ety", "revg",
              "ocfC", "fcfC", "one"]

def feats_vec(y, c):
    g5 = c["g5"]; ety = c.get("eps_this_y") or 0.0; eny = c.get("eps_next_y") or 0.0
    return np.array([g5, np.sqrt(max(g5, 0)), eny, np.sqrt(abs(eny)) * np.sign(eny),
                     ety, np.sqrt(abs(ety)) * np.sign(ety),
                     (y.get("revenueGrowth") or 0) * 100,
                     cagr(y.get("ocf_hist")) or 0.0, cagr(y.get("fcf_hist")) or 0.0, 1.0])

def bases_of(y):
    ocf = y.get("ocf_ttm_q") or y["ocf_ttm"]
    fcf = y.get("fcf_ttm_q") or y["fcf_ttm_yahoo"]
    caps = [abs(x) for x in (y.get("capex_hist") or []) if x is not None]
    fcfavg = ocf - sum(caps) / len(caps) if caps else None
    return dict(ocf=ocf, fcf=fcf, fcfavg=fcfavg, ni=y.get("ni_ttm"))

D = {}
for t, y in Y.items():
    c = C[t]
    sh = y["shares"]; disc = RF + (y["beta"] or 1.0) * MRP
    bases = bases_of(y)
    implied = {}
    for bn, bv in bases.items():
        if not bv or bv <= 0:
            continue
        try:
            implied[bn] = brentq(
                lambda g: iv(bv / sh, g, disc, y["totalDebt"]/sh, y["totalCash"]/sh) - c["target_iv"],
                -0.6, 2.5) * 100
        except Exception:
            pass
    D[t] = dict(x=feats_vec(y, c), implied=implied, sh=sh, disc=disc,
                debt_ps=y["totalDebt"]/sh, cash_ps=y["totalCash"]/sh,
                bases=bases, tgt=c["target_iv"])

tickers = list(D)
picks = {t: min(D[t]["implied"], key=lambda b: abs(D[t]["implied"][b] - 12.0)) for t in tickers}

for it in range(20):
    X = np.array([D[t]["x"] for t in tickers])
    gvec = np.array([D[t]["implied"][picks[t]] for t in tickers])
    w, *_ = np.linalg.lstsq(X, gvec, rcond=None)
    new_picks = {}
    for t in tickers:
        pred = float(D[t]["x"] @ w)
        new_picks[t] = min(D[t]["implied"], key=lambda b: abs(D[t]["implied"][b] - pred))
    if new_picks == picks:
        break
    picks = new_picks

X = np.array([D[t]["x"] for t in tickers])
gvec = np.array([D[t]["implied"][picks[t]] for t in tickers])
w, *_ = np.linalg.lstsq(X, gvec, rcond=None)
print("weights:", {n: round(float(v), 4) for n, v in zip(FEAT_NAMES, w)})
errs = {}
for t in tickers:
    d = D[t]
    g = float(d["x"] @ w) / 100.0
    bv = d["bases"][picks[t]]
    v = iv(bv / d["sh"], g, d["disc"], d["debt_ps"], d["cash_ps"])
    errs[t] = (v - d["tgt"]) / d["tgt"] * 100
    print(f"  {t:6} g={g*100:6.2f}%  base={picks[t]:7}  err={errs[t]:+6.2f}%")
mx = max(abs(e) for e in errs.values())
print(f"max|err| = {mx:.2f}%   MAPE = {np.mean([abs(e) for e in errs.values()]):.2f}%")
json.dump(dict(feat_names=FEAT_NAMES, weights=list(map(float, w)), picks=picks,
               errs=errs, max_abs_err=mx),
          open(os.path.join(HERE, "blend_params.json"), "w"), indent=1)
print("saved blend_params.json")
