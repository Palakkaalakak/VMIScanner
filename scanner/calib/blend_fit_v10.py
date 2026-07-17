#!/usr/bin/env python3
"""v10: combine (a) deterministic push + (c) continuous per-sector flow mix,
now over a WIDE component set: SEC annual (ocf,fcf,ni), finviz fwdni,
stockanalysis TTM (ocfT,fcfT,niT), 3y SEC averages (ocf3,fcf3,ni3).
Alternating optimization:
  step G: fix sector alphas -> per-ticker base -> growth interval [gLo,gHi]
          -> LP for growth weights (features + sector intercepts), min slack
  step A: fix growth weights -> per-ticker required base interval -> per-sector
          LP over alphas (>=0, sum=1), min slack
Multi-seed, then +/-5% center-pull polish. Excl: NVO, BABA (FX), ACGL
(insurance, production-excluded), CNSWF/EVVTY/LVMUY (no data).
"""
import json, math
import numpy as np
from scipy.optimize import linprog

RF, MRP = 3.608, 2.728
TOL = 7.0
EXCL = {'NVO', 'BABA', 'ACGL', 'CNSWF', 'EVVTY', 'LVMUY'}
COMPS = ['ocf', 'fcf', 'ni', 'fwdni', 'ocfT', 'fcfT', 'niT', 'ocf3', 'fcf3', 'ni3']
FALLBACK = {'ocfT': 'ocf', 'fcfT': 'fcf', 'niT': 'ni',
            'ocf3': 'ocf', 'fcf3': 'fcf', 'ni3': 'ni', 'fwdni': 'ni'}

inp = json.load(open('inputs2.json'))
sa = json.load(open('sa_inputs.json'))


def pv_coef(g_pct, disc_pct):
    """PV of 20y flow stream per unit of base flow (g yrs1-10, 4% yrs11-20)."""
    g, r = g_pct / 100.0, disc_pct / 100.0
    f, pv = 1.0, 0.0
    for yr in range(1, 21):
        f *= 1 + (g if yr <= 10 else 0.04)
        pv += f / (1 + r) ** yr
    return pv


def solve_g(bp, disc, net, target):
    """g such that bp*pv_coef(g)+net == target (net = cash_ps - debt_ps)."""
    lo, hi = -60.0, 300.0
    def iv(g): return bp * pv_coef(g, disc) + net
    if iv(hi) < target or iv(lo) > target:
        return None
    for _ in range(70):
        mid = (lo + hi) / 2
        if iv(mid) < target: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def sg(x): return math.copysign(math.sqrt(abs(x)), x)


def avg3(series):
    vals = [v for v in (series or [])[:3] if v is not None]
    return sum(vals) / len(vals) if len(vals) >= 2 else None


# ---------- build ticker table ----------
SECTORS = sorted({d['sector'] for t, d in inp.items() if t not in EXCL})
T = []
for t, d in sorted(inp.items()):
    if t in EXCL: continue
    fv, sec = d.get('finviz'), d.get('sec')
    if not fv or not sec: continue
    sh = fv['shares_outstanding']; g5 = fv['eps_next_5y']
    eny, ety = fv['eps_next_y'], fv['eps_this_y']
    beta = fv.get('beta') or 1.0
    disc = RF + beta * MRP
    tgt = d['target_iv']
    debt = ((sec.get('std') or 0) + (sec.get('ltd') or 0)) / sh
    cash = ((sec.get('cash') or 0) + (sec.get('sti') or 0)) / sh
    net = cash - debt
    ocf = sec['ocf'][0]
    capex = abs(sec['capex'][0]) if sec['capex'] and sec['capex'][0] is not None else 0.0
    ni = sec['netIncome'][0]
    flows = {'ocf': ocf,
             'fcf': (ocf - capex) if ocf is not None else None,
             'ni': ni,
             'fwdni': (fv.get('fwd_eps') or 0) * sh or None,
             'ocf3': avg3(sec['ocf']), 'ni3': avg3(sec['netIncome'])}
    fcfs = [(o - abs(c)) if (o is not None and c is not None) else None
            for o, c in zip(sec['ocf'], sec['capex'])]
    flows['fcf3'] = avg3(fcfs)
    s = sa.get(t)
    if s and s.get('ttm0'):
        oT, cT, nT = s['ocf'][0], s['capex'][0], s['ni'][0]
        flows['ocfT'] = oT
        flows['fcfT'] = oT + cT  # capex negative in sa
        flows['niT'] = nT
    # fallback chain + positivity; per-share
    fps = {}
    for c in COMPS:
        v = flows.get(c)
        if v is None or v <= 0:
            v = flows.get(FALLBACK.get(c, c))
        fps[c] = (v / sh) if (v is not None and v > 0) else 0.0
    # growth features
    ocfC = None
    so = sec['ocf']
    if len(so) >= 6 and so[5] and so[5] > 0 and so[0] and so[0] > 0:
        ocfC = ((so[0] / so[5]) ** 0.2 - 1) * 100
    fcfC = None
    if len(fcfs) >= 6 and fcfs[5] and fcfs[5] > 0 and fcfs[0] and fcfs[0] > 0:
        fcfC = ((fcfs[0] / fcfs[5]) ** 0.2 - 1) * 100
    feats = [g5, sg(g5), eny, sg(eny), ety, sg(ety), ocfC or 0.0, fcfC or 0.0, 1.0]
    sd = [1.0 if d['sector'] == s2 else 0.0 for s2 in SECTORS]
    T.append(dict(t=t, sector=d['sector'], tgt=tgt, disc=disc, net=net,
                  fps=fps, x=np.array(feats + sd), sog=d['so_growth'], g5=g5))

N = len(T)
NF = len(T[0]['x'])
print(f'{N} tickers, {NF} features, sectors: {SECTORS}')


def base_ps(tk, alpha):
    a = alpha[tk['sector']]
    return sum(a[i] * tk['fps'][c] for i, c in enumerate(COMPS))


def g_interval(tk, bp, tol):
    lo = solve_g(bp, tk['disc'], tk['net'], tk['tgt'] * (1 - tol / 100))
    hi = solve_g(bp, tk['disc'], tk['net'], tk['tgt'] * (1 + tol / 100))
    if lo is None or hi is None: return None
    return (lo, hi)


def fit_growth_lp(ivls):
    """min sum slack s.t. gLo-s <= w.x <= gHi+s ; w free."""
    rows_ub, b_ub = [], []
    idx = [i for i, iv in enumerate(ivls) if iv is not None]
    ns = len(idx)
    # vars: w (NF, free -> wp-wn) + slack(ns)
    nv = 2 * NF + ns
    cvec = np.zeros(nv); cvec[2 * NF:] = 1.0
    for k, i in enumerate(idx):
        lo, hi = ivls[i]; x = T[i]['x']
        r = np.zeros(nv); r[:NF] = x; r[NF:2 * NF] = -x; r[2 * NF + k] = -1
        rows_ub.append(r); b_ub.append(hi)
        r = np.zeros(nv); r[:NF] = -x; r[NF:2 * NF] = x; r[2 * NF + k] = -1
        rows_ub.append(r); b_ub.append(-lo)
    res = linprog(cvec, A_ub=np.array(rows_ub), b_ub=np.array(b_ub),
                  bounds=[(0, None)] * nv, method='highs')
    if not res.success: return None
    w = res.x[:NF] - res.x[NF:2 * NF]
    return w


def fit_alpha_lp(w):
    """per sector: alphas>=0 sum=1, min slack of base within required interval."""
    alpha = {}
    for s in SECTORS:
        sk = [tk for tk in T if tk['sector'] == s]
        nc, ns = len(COMPS), len(sk)
        nv = nc + ns
        cvec = np.zeros(nv); cvec[nc:] = 1.0
        A_ub, b_ub = [], []
        for k, tk in enumerate(sk):
            g = float(np.dot(w, tk['x']))
            g = max(-12.0, min(80.0, g))
            pc = pv_coef(g, tk['disc'])
            blo = (tk['tgt'] * (1 - TOL / 100) - tk['net']) / pc
            bhi = (tk['tgt'] * (1 + TOL / 100) - tk['net']) / pc
            f = np.array([tk['fps'][c] for c in COMPS])
            r = np.zeros(nv); r[:nc] = f; r[nc + k] = -abs(bhi) - 1e-9
            A_ub.append(r); b_ub.append(bhi)
            r = np.zeros(nv); r[:nc] = -f; r[nc + k] = -abs(bhi) - 1e-9
            A_ub.append(r); b_ub.append(-blo)
        A_eq = np.zeros((1, nv)); A_eq[0, :nc] = 1.0
        res = linprog(cvec, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * nv,
                      method='highs')
        if res.success:
            alpha[s] = res.x[:nc]
        else:
            alpha[s] = np.array([0, 0, 1.0] + [0] * (nc - 3))  # ni fallback
    return alpha


def evaluate(w, alpha, tol=TOL):
    errs, hits = {}, 0
    for tk in T:
        g = max(-12.0, min(80.0, float(np.dot(w, tk['x']))))
        bp = base_ps(tk, alpha)
        iv = bp * pv_coef(g, tk['disc']) + tk['net']
        e = (iv / tk['tgt'] - 1) * 100
        errs[tk['t']] = e
        if abs(e) <= tol: hits += 1
    return hits, errs


# ---------- seeds ----------
def seed_alpha(kind):
    a = {}
    for s in SECTORS:
        v = np.zeros(len(COMPS))
        if kind == 'ni': v[COMPS.index('ni')] = 1
        elif kind == 'ocf': v[COMPS.index('ocf')] = 1
        elif kind == 'fcf': v[COMPS.index('fcf')] = 1
        elif kind == 'ttm': v[COMPS.index('ocfT')] = 0.5; v[COMPS.index('niT')] = 0.5
        elif kind == 'mix': v[:] = 1.0 / len(COMPS)
        elif kind == 'v9':
            m = {'fin-net': {'fcf': 1.0}, 'health': {'ni': 1.0},
                 'industrial': {'fwdni': 0.78, 'fcf': 0.22},
                 'consumer': {'ni': 0.62, 'fcf': 0.38},
                 'retail': {'ni': 0.62, 'fcf': 0.38},
                 'tech-sw': {'ocf': 0.5, 'ni': 0.5},
                 'tech-net': {'fcf': 0.6, 'fwdni': 0.4},
                 'tech-hw': {'ni': 1.0}, 'insurance': {'ni': 1.0}}
            for c, wt in m.get(s, {'ni': 1.0}).items():
                v[COMPS.index(c)] = wt
        a[s] = v
    return a


best = None
for kind in ['v9', 'mix', 'ni', 'ocf', 'fcf', 'ttm']:
    alpha = seed_alpha(kind)
    w = None
    for it in range(14):
        ivls = [g_interval(tk, base_ps(tk, alpha), TOL) for tk in T]
        w2 = fit_growth_lp(ivls)
        if w2 is None: break
        w = w2
        alpha = fit_alpha_lp(w)
    if w is None: continue
    hits, errs = evaluate(w, alpha)
    print(f'seed={kind:4s}  hits ±7%: {hits}/{N}  maxerr={max(abs(e) for e in errs.values()):.1f}')
    if best is None or hits > best[0]:
        best = (hits, w, {s: alpha[s].copy() for s in alpha}, errs)

hits, w, alpha, errs = best
print(f'\n=== BEST: {hits}/{N} within ±7% ===')

# ---------- ±5% polish: re-run alternation with tighter tol on hits ----------
w5, a5 = w.copy(), {s: alpha[s].copy() for s in alpha}
for it in range(8):
    ivls = []
    for tk in T:
        bp = base_ps(tk, a5)
        iv5 = g_interval(tk, bp, 5.0)
        ivls.append(iv5 if iv5 else g_interval(tk, bp, TOL))
    w2 = fit_growth_lp(ivls)
    if w2 is None: break
    w5 = w2
    a5 = fit_alpha_lp(w5)
h5, e5 = evaluate(w5, a5)
h5in5 = sum(1 for e in e5.values() if abs(e) <= 5.0)
print(f'polished: {h5}/{N} within ±7%, {h5in5}/{N} within ±5%')
if h5 >= hits:
    w, alpha, errs, hits = w5, a5, e5, h5

in5 = sum(1 for e in errs.values() if abs(e) <= 5.0)
print(f'\nFINAL: {hits}/{N} ±7%  |  {in5}/{N} ±5%')
print('\nper-ticker errors (sorted by |err|):')
for t, e in sorted(errs.items(), key=lambda kv: -abs(kv[1])):
    sec = next(tk['sector'] for tk in T if tk['t'] == t)
    mark = '' if abs(e) <= 5 else (' *' if abs(e) <= 7 else ' **MISS**')
    print(f'  {t:6s} {sec:11s} {e:+7.2f}%{mark}')

print('\nsector alphas (weights >1%):')
for s in SECTORS:
    aw = {c: round(float(v), 3) for c, v in zip(COMPS, alpha[s]) if v > 0.01}
    print(f'  {s:11s} {aw}')

print('\ngrowth weights:')
names = ['g5', 'sqrt_g5', 'eny', 'sqrt_eny', 'ety', 'sqrt_ety', 'ocfC', 'fcfC', 'const'] + ['sec_' + s for s in SECTORS]
for n, v in zip(names, w):
    print(f'  {n:14s} {v:+9.4f}')

json.dump({'weights': list(map(float, w)), 'feature_names': names,
           'sectors': SECTORS, 'components': COMPS,
           'alphas': {s: list(map(float, alpha[s])) for s in SECTORS},
           'hits7': hits, 'hits5': in5, 'n': N,
           'errors': {t: round(e, 3) for t, e in errs.items}},
          open('fit_v10.json', 'w'), indent=1) if False else None
with open('fit_v10.json', 'w') as f:
    json.dump({'weights': list(map(float, w)), 'feature_names': names,
               'sectors': SECTORS, 'components': COMPS,
               'alphas': {s: list(map(float, alpha[s])) for s in SECTORS},
               'hits7': hits, 'hits5': in5, 'n': N,
               'errors': {t: round(e, 3) for t, e in errs.items()}}, f, indent=1)
print('\nsaved fit_v10.json')
