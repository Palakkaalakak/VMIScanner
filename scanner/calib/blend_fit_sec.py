"""Consolidated SEC-native blend fit (recreated after sandbox reset).

Goal: reproduce StockOracle Base IV on 11 benchmarks within ±4% using ONLY
data the production scanner has at scan time (SEC latest-FY flows/balance,
finviz estimates/beta/shares), under the verified DCF20 structure
(growth yrs 1-10, 4% yrs 11-20, CAPM discount RF 3.608% + beta·2.728%,
IV = PV/sh − debt/sh + cash/sh). No caps, no minimums, no per-ticker memory.

Two coupled unknowns:
  1. base-flow selection (ocf | fcf | ocf−5yAvgCapex | ni) — must be a
     DETERMINISTIC rule usable for arbitrary tickers;
  2. growth-blend weights g = w·features — must extrapolate sanely
     (monotone-ish in analyst 5y growth, no explosions at the tails).

Search: enumerate two-feature threshold trees over (capex/ocf, g5)
   leaf0: capex_ocf <= a           -> base L0
   leaf1: else if g5 <= tg         -> base L1
   leaf2: else if capex_ocf <= b   -> base L2
   leaf3: else if capex_ocf <= c   -> base L3
   leaf4: else                     -> base L4
(degenerate trees allowed by merging thresholds), least-squares fit the
weights on the rule-implied g targets, keep candidates meeting both the
error and the sanity gates, prefer fewest leaves then lowest max error.

Prior session results (lost to reset, for reference):
  v8  single EM on SEC data ............ max|err| 3.82%
  v9  multi-restart EM ................. 0.11% but INSANE weights (210% @ g5=0)
  v10 interval-QP + sqrt feats ......... 3.40%, picks {AAPL:ocf, AMZN:ocf,
      GOOGL:fcfavg, MA:ni, META:fcf, MSFT:ocf, NVDA:ni, PANW:ocf, SPGI:ni,
      TMO:ni, WM:fcfavg}
  v11 exact LS on v10 picks ............ 3.65%, sane sweep 6.0/13.9/34.6/56.1
  tree on v10 picks .................... only 10/11 (AMZN misrouted)
  v12 single-feature rules @3.5% ....... 0 hits  -> hence 2-feature family here
"""
import itertools, json, os, sys
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(__file__)
S = json.load(open(os.path.join(HERE, "sec_series.json")))
C = json.load(open(os.path.join(HERE, "calib_inputs.json")))
RF, MRP = 0.03608, 0.02728
BASES = ("ocf", "fcf", "fcfavg", "ni")
TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 3.5


def cagr(seq, n=None):
    seq = [x for x in (seq or []) if x is not None]
    if n:
        seq = seq[:n]
    if len(seq) >= 3 and seq[0] > 0 and seq[-1] > 0:
        return ((seq[0] / seq[-1]) ** (1 / (len(seq) - 1)) - 1) * 100
    return 0.0


def iv(base_ps, g, disc, debt_ps, cash_ps):
    pv, f = 0.0, base_ps
    for yr in range(1, 21):
        f *= 1 + (g if yr <= 10 else 0.04)
        pv += f / (1 + disc) ** yr
    return pv - debt_ps + cash_ps


FEAT_NAMES = ["g5", "sqrt_g5", "eny", "sqrt_eny", "ety", "sqrt_ety",
              "ocfC", "fcfC", "one"]


def feats_vec(s, c):
    g5 = c["g5"]; ety = c.get("eps_this_y") or 0.0; eny = c.get("eps_next_y") or 0.0
    return np.array([g5, np.sqrt(max(g5, 0)),
                     eny, np.sqrt(abs(eny)) * np.sign(eny),
                     ety, np.sqrt(abs(ety)) * np.sign(ety),
                     cagr(s.get("ocf"), 6), cagr(s.get("fcf"), 6), 1.0])


D, RF_FEATS = {}, {}
for t, s in S.items():
    c = C[t]
    sh = c["shares"]; disc = RF + (c.get("beta") or 1.0) * MRP
    debt = (s.get("std") or 0) + (s.get("ltd") or 0)
    cash = (s.get("cash") or 0) + (s.get("sti") or 0)
    ocf = (s.get("ocf") or [None])[0]
    fcf = (s.get("fcf") or [None])[0]
    caps = [abs(x) for x in (s.get("capex") or [])[:5] if x]
    fcfavg = (ocf - sum(caps) / len(caps)) if (ocf and caps) else None
    ni = (s.get("netIncome") or [None])[0]
    bases = dict(ocf=ocf, fcf=fcf, fcfavg=fcfavg, ni=ni)
    implied = {}
    for bn, bv in bases.items():
        if not bv or bv <= 0:
            continue
        try:
            implied[bn] = brentq(
                lambda g: iv(bv / sh, g, disc, debt / sh, cash / sh) - c["target_iv"],
                -0.6, 2.5) * 100
        except Exception:
            pass
    D[t] = dict(x=feats_vec(s, c), implied=implied, sh=sh, disc=disc,
                debt_ps=debt / sh, cash_ps=cash / sh, bases=bases,
                tgt=c["target_iv"])
    RF_FEATS[t] = dict(
        capex_ocf=(caps[0] / ocf * 100) if caps and ocf else 0.0,
        g5=c["g5"])

tickers = sorted(D)
Xall = np.array([D[t]["x"] for t in tickers])


def eval_picks(picks):
    if any(picks[t] not in D[t]["implied"] for t in tickers):
        return None, None, np.inf
    gvec = np.array([D[t]["implied"][picks[t]] for t in tickers])
    w, *_ = np.linalg.lstsq(Xall, gvec, rcond=None)
    errs = {}
    for t in tickers:
        d = D[t]
        g = float(d["x"] @ w) / 100.0
        v = iv(d["bases"][picks[t]] / d["sh"], g, d["disc"], d["debt_ps"], d["cash_ps"])
        errs[t] = (v - d["tgt"]) / d["tgt"] * 100
    return w, errs, max(abs(e) for e in errs.values())


def gpred(w, g5, eny=12, ety=10, ocfC=10, fcfC=10):
    x = np.array([g5, np.sqrt(max(g5, 0)), eny, np.sqrt(abs(eny)) * np.sign(eny),
                  ety, np.sqrt(abs(ety)) * np.sign(ety), ocfC, fcfC, 1.0])
    return float(x @ w)


def sane(w):
    vals = [gpred(w, g5) for g5 in (0, 5, 10, 20, 30, 50)]
    if not all(vals[i] <= vals[i + 1] + 3 for i in range(len(vals) - 1)):
        return False
    return -15 < vals[0] and vals[-1] < 80 and all(-30 < v < 100 for v in vals)


def rule_picks(a, tg, b, cthr, leaves):
    L0, L1, L2, L3, L4 = leaves
    out = {}
    for t in tickers:
        co, g5 = RF_FEATS[t]["capex_ocf"], RF_FEATS[t]["g5"]
        if co <= a:
            out[t] = L0
        elif g5 <= tg:
            out[t] = L1
        elif co <= b:
            out[t] = L2
        elif co <= cthr:
            out[t] = L3
        else:
            out[t] = L4
    return out


cap_vals = sorted(set(RF_FEATS[t]["capex_ocf"] for t in tickers))
g5_vals = sorted(set(RF_FEATS[t]["g5"] for t in tickers))
cap_mids = [(cap_vals[i] + cap_vals[i + 1]) / 2 for i in range(len(cap_vals) - 1)]
g5_mids = [(g5_vals[i] + g5_vals[i + 1]) / 2 for i in range(len(g5_vals) - 1)]
BIG = 1e9  # sentinel to collapse a threshold (degenerate = simpler tree)

results = []
seen = set()
cap_choices_b = cap_mids + [BIG]
for a in cap_mids:
    for tg in g5_mids + [-BIG]:
        for b in cap_choices_b:
            if b <= a and b != BIG:
                continue
            for cthr in cap_choices_b:
                if cthr != BIG and (cthr <= max(a, b if b != BIG else a)):
                    continue
                for leaves in itertools.product(BASES, repeat=5):
                    picks = rule_picks(a, tg, b, cthr, leaves)
                    key = tuple(picks[t] for t in tickers)
                    if key in seen:
                        continue
                    seen.add(key)
                    w, errs, mx = eval_picks(picks)
                    if mx <= TOL and sane(w):
                        nleaf = len(set(key))
                        results.append((mx, nleaf, a, tg, b, cthr, leaves,
                                        picks, w, errs))

results.sort(key=lambda r: (r[0]))
print(f"{len(results)} sane rules with max|err| <= {TOL}%  "
      f"({len(seen)} distinct assignments tried)")
for r in results[:12]:
    mx, nleaf, a, tg, b, cthr, leaves = r[:7]
    print(f"  max|err|={mx:5.2f}%  a={a:6.2f} tg={tg:7.2f} b={b if b != BIG else -1:7.2f} "
          f"c={cthr if cthr != BIG else -1:7.2f}  leaves={leaves}")

if results:
    mx, nleaf, a, tg, b, cthr, leaves, picks, w, errs = results[0]
    print("\nWINNER rule:")
    print(f"  capex/ocf <= {a:.2f}%%              -> {leaves[0]}")
    print(f"  elif g5 <= {tg:.2f}                -> {leaves[1]}")
    print(f"  elif capex/ocf <= {b if b != BIG else 999:.2f}        -> {leaves[2]}")
    print(f"  elif capex/ocf <= {cthr if cthr != BIG else 999:.2f}  -> {leaves[3]}")
    print(f"  else                              -> {leaves[4]}")
    print("  weights:", {n: round(float(v), 4) for n, v in zip(FEAT_NAMES, w)})
    for t in tickers:
        g = float(D[t]["x"] @ w)
        print(f"  {t:6} g={g:6.2f}%  base={picks[t]:7}  err={errs[t]:+6.2f}%")
    print(f"  max|err| = {mx:.2f}%")
    print("  sanity g5 sweep:", [round(gpred(w, g5), 1) for g5 in (0, 5, 10, 20, 30, 50)])
    json.dump(dict(feat_names=FEAT_NAMES, weights=list(map(float, w)),
                   rule=dict(a=a, tg=tg, b=(None if b == BIG else b),
                             c=(None if cthr == BIG else cthr),
                             leaves=list(leaves)),
                   picks=picks, errs=errs, max_abs_err=mx),
              open(os.path.join(HERE, "blend_params_sec.json"), "w"), indent=1)
    print("saved blend_params_sec.json")
