#!/usr/bin/env python3
"""v2 joint calibration on the expanded 42-ticker StockOracle benchmark set.

Strategy:
  1. Per ticker & base-flow candidate (ocf/fcf/ni/fwdni), solve the growth
     interval [gLo,gHi] such that IV lands within +/-TOL of Base IV
     (verified DCF20 structure + CAPM discount).
  2. Coordinate-descent: alternate (a) fit growth-blend weights by interval
     least squares (EM projection) and (b) per-ticker pick the base whose
     interval is nearest the current prediction.
  3. Distill the resulting picks into a deterministic decision tree over
     public ratios (capex/OCF, g5, ni/ocf, ...) so production has no lookup.
  4. Sanity gates: monotone in g5, bounded predictions at feature extremes.

Excluded (justified): NVO, BABA (SEC facts in home currency: DKK/CNY, target
IVs in USD -> flows off by FX factor); CNSWF/EVVTY/LVMUY (no SEC/finviz data).
ACGL kept but reported separately (insurance; no base reproduces SO's IV).
"""
import json, math, itertools, sys
import numpy as np

RF, MRP = 3.608, 2.728
TOL_IDEAL, TOL_MAX = 5.0, 7.0
EXCLUDE_FX = {'NVO', 'BABA'}
BASES = ('ocf', 'fcf', 'ni', 'fwdni')

inp = json.load(open('inputs2.json'))

def cagrN(series, n):
    s = series or []
    if len(s) < n+1: return None
    new, old = s[0], s[n]
    if new is None or old is None or old <= 0 or new <= 0: return None
    return ((new/old)**(1.0/n)-1)*100

def dcf(base_ps, g_pct, disc_pct, debt_ps, cash_ps):
    g, r = g_pct/100.0, disc_pct/100.0
    f, pv = base_ps, 0.0
    for yr in range(1, 21):
        f *= 1 + (g if yr <= 10 else 0.04)
        pv += f/(1+r)**yr
    return pv - debt_ps + cash_ps

def solve_g(base_ps, disc, debt_ps, cash_ps, target):
    lo, hi = -60.0, 300.0
    if dcf(base_ps, hi, disc, debt_ps, cash_ps) < target: return None
    if dcf(base_ps, lo, disc, debt_ps, cash_ps) > target: return None
    for _ in range(80):
        mid = (lo+hi)/2
        if dcf(base_ps, mid, disc, debt_ps, cash_ps) < target: lo = mid
        else: hi = mid
    return (lo+hi)/2

def sgnsqrt(x):
    return math.copysign(math.sqrt(abs(x)), x)

# ---------------- build dataset ----------------
FEATS = ['g5','sq_g5','eny','sq_eny','ety','sq_ety','ocfC','fcfC','revC','one']
data = {}
for t, d in sorted(inp.items()):
    fv, sec = d.get('finviz'), d.get('sec')
    if not fv or not sec or t in EXCLUDE_FX: continue
    sh = fv.get('shares_outstanding')
    g5, eny, ety = fv.get('eps_next_5y'), fv.get('eps_next_y'), fv.get('eps_this_y')
    if not sh or g5 is None or eny is None or ety is None: continue
    beta = fv.get('beta') or 1.0
    disc = RF + beta*MRP
    tgt = d['target_iv']
    ocf = sec['ocf'][0] if sec['ocf'] else None
    capex = abs(sec['capex'][0]) if sec['capex'] and sec['capex'][0] is not None else None
    ni = sec['netIncome'][0] if sec['netIncome'] else None
    fcf = (ocf-capex) if (ocf is not None and capex is not None) else None
    fwd = fv.get('fwd_eps')
    fwdni = fwd*sh if fwd else None
    debt = ((sec.get('std') or 0)+(sec.get('ltd') or 0))/sh
    cash = ((sec.get('cash') or 0)+(sec.get('sti') or 0))/sh
    fcfs = [(o-abs(c)) if (o is not None and c is not None) else None
            for o, c in zip(sec['ocf'], sec['capex'])]
    ocfC = cagrN(sec['ocf'], 5) or 0.0
    fcfC = cagrN(fcfs, 5) or 0.0
    revC = cagrN(sec['revenue'], 5) or 0.0
    x = np.array([g5, sgnsqrt(g5), eny, sgnsqrt(eny), ety, sgnsqrt(ety),
                  ocfC, fcfC, revC, 1.0])
    # ratio features for the rule tree
    co = (capex/ocf*100) if (capex is not None and ocf and ocf > 0) else None
    ni_ocf = (ni/ocf*100) if (ni is not None and ocf and ocf > 0) else None
    ivl = {}
    for name, flow in (('ocf',ocf),('fcf',fcf),('ni',ni),('fwdni',fwdni)):
        if flow is None or flow <= 0: continue
        bp = flow/sh
        lo7 = solve_g(bp, disc, debt, cash, tgt*(1-TOL_MAX/100))
        hi7 = solve_g(bp, disc, debt, cash, tgt*(1+TOL_MAX/100))
        lo5 = solve_g(bp, disc, debt, cash, tgt*(1-TOL_IDEAL/100))
        hi5 = solve_g(bp, disc, debt, cash, tgt*(1+TOL_IDEAL/100))
        gc  = solve_g(bp, disc, debt, cash, tgt)
        if lo7 is None or hi7 is None: continue
        ivl[name] = {'lo7':lo7,'hi7':hi7,'lo5':lo5,'hi5':hi5,'g':gc,
                     'flow_ps':bp}
    if not ivl: continue
    data[t] = {'x':x, 'ivl':ivl, 'sector':d['sector'], 'tgt':tgt,
               'disc':disc, 'debt':debt, 'cash':cash, 'sog':d['so_growth'],
               'co':co, 'ni_ocf':ni_ocf, 'g5':g5}

tickers = sorted(data)
print(f"fit set: {len(tickers)} tickers -> {tickers}")

X = np.vstack([data[t]['x'] for t in tickers])

def fit_interval_ls(picks, ridge=1e-3, iters=60):
    """EM interval regression: weights w s.t. X.w lands inside per-ticker
    [lo5,hi5] intervals (fall back to lo7/hi7 bounds as hard-ish)."""
    lo = np.array([data[t]['ivl'][picks[t]]['lo5'] if data[t]['ivl'][picks[t]]['lo5'] is not None
                   else data[t]['ivl'][picks[t]]['lo7'] for t in tickers])
    hi = np.array([data[t]['ivl'][picks[t]]['hi5'] if data[t]['ivl'][picks[t]]['hi5'] is not None
                   else data[t]['ivl'][picks[t]]['hi7'] for t in tickers])
    y = (lo+hi)/2
    w = None
    for _ in range(iters):
        A = np.vstack([X, math.sqrt(ridge)*np.eye(X.shape[1])])
        b = np.concatenate([y, np.zeros(X.shape[1])])
        w, *_ = np.linalg.lstsq(A, b, rcond=None)
        p = X @ w
        y = np.clip(p, lo, hi)
    return w

def eval_weights(w, picks):
    errs = {}
    for i, t in enumerate(tickers):
        g = float(X[i] @ w)
        d = data[t]; b = picks[t]
        iv = dcf(d['ivl'][b]['flow_ps'], g, d['disc'], d['debt'], d['cash'])
        errs[t] = (iv/d['tgt']-1)*100
    return errs

def sanity(w):
    """predictions must be monotone-ish in g5 and bounded"""
    base = np.median(X, axis=0)
    prev = None
    for g5v in (0, 5, 10, 20, 40, 60, 80):
        x = base.copy(); x[0] = g5v; x[1] = math.sqrt(g5v)
        p = float(x @ w)
        if p < -5 or p > 70: return False
        if prev is not None and p < prev - 1.5: return False
        prev = p
    return True

# ------------- coordinate descent over picks -------------
# init: pick base whose center implied g is closest to finviz g5 (a public prior)
picks = {}
for t in tickers:
    best = min(data[t]['ivl'].items(),
               key=lambda kv: abs(kv[1]['g'] - data[t]['g5']))
    picks[t] = best[0]

best_state = None
for it in range(30):
    w = fit_interval_ls(picks)
    errs = eval_weights(w, picks)
    mx = max(abs(e) for e in errs.values())
    n7 = sum(1 for e in errs.values() if abs(e) <= TOL_MAX)
    n5 = sum(1 for e in errs.values() if abs(e) <= TOL_IDEAL)
    ok = sanity(w)
    if best_state is None or (n7, n5, -mx) > (best_state[0], best_state[1], -best_state[2]):
        best_state = (n7, n5, mx, w.copy(), dict(picks), dict(errs))
    # reassign picks: base whose interval center is nearest current prediction
    changed = 0
    for i, t in enumerate(tickers):
        g = float(X[i] @ w)
        cands = data[t]['ivl']
        def dist(b):
            lo, hi = cands[b]['lo7'], cands[b]['hi7']
            if lo <= g <= hi: return 0.0
            return min(abs(g-lo), abs(g-hi))
        nb = min(cands, key=dist)
        if nb != picks[t]:
            picks[t] = nb; changed += 1
    if changed == 0:
        break

n7, n5, mx, w, picks, errs = best_state
print(f"\ncoordinate-descent result: within±7%: {n7}/{len(tickers)}  within±5%: {n5}  max|e|={mx:.2f}  sane={sanity(w)}")
print('weights:', {f: round(float(v),4) for f, v in zip(FEATS, w)})
print('\npicks + errors:')
from collections import defaultdict
bysec = defaultdict(list)
for t in tickers:
    e = errs[t]
    flag = '' if abs(e) <= 5 else (' *' if abs(e) <= 7 else ' <<<')
    print(f"  {t:7s}{data[t]['sector']:11s} base={picks[t]:6s} err={e:+7.2f}{flag}")
    bysec[data[t]['sector']].append(e)
import statistics as st
print('\nper-sector residuals:')
for s, es in sorted(bysec.items()):
    print(f"  {s:11s} n={len(es):2d} mean={st.mean(es):+6.2f} med={st.median(es):+6.2f} max|e|={max(abs(x) for x in es):6.2f}")

json.dump({'feat_names': FEATS, 'weights': [float(v) for v in w],
           'picks': picks, 'errs': errs,
           'ticker_features': {t: {'co': data[t]['co'], 'g5': data[t]['g5'],
                                   'ni_ocf': data[t]['ni_ocf']} for t in tickers}},
          open('fit_v2_stage1.json','w'), indent=1)
print('\nwrote fit_v2_stage1.json (weights + free picks; rule distillation next)')
