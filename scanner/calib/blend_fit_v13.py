#!/usr/bin/env python3
"""v12 = v11 + fundamental separators (net margin, capex/OCF, rev 5y CAGR)
added as continuous growth features — no caps/minimums, just more signal: (a)+(c) combo over wide component set with
 - inner fit tolerance (fit at ±6.2%, evaluate at ±7%) to avoid boundary-riding
 - best-iterate tracking across alternation (eval every step)
 - center-pull growth refinement (pull toward interval centers of feasible set)
 - per-ticker keep-set weighting: tickers currently hit get hard constraints,
   misses get slack -> greedy count maximization instead of sum-slack.
"""
import json, math
import numpy as np
from scipy.optimize import linprog

RF, MRP = 3.608, 2.728
TOL = 7.0
TOL_FIT = 6.2
EXCL = {'NVO', 'BABA', 'ACGL', 'CNSWF', 'EVVTY', 'LVMUY'}
COMPS = ['ocf', 'fcf', 'ni', 'fwdni', 'ocfT', 'fcfT', 'niT', 'ocf3', 'fcf3', 'ni3']
FALLBACK = {'ocfT': 'ocf', 'fcfT': 'fcf', 'niT': 'ni',
            'ocf3': 'ocf', 'fcf3': 'fcf', 'ni3': 'ni', 'fwdni': 'ni'}

inp = json.load(open('inputs2.json'))
sa = json.load(open('sa_inputs.json'))


def pv_coef(g_pct, disc_pct):
    g, r = g_pct / 100.0, disc_pct / 100.0
    f, pv = 1.0, 0.0
    for yr in range(1, 21):
        f *= 1 + (g if yr <= 10 else 0.04)
        pv += f / (1 + r) ** yr
    return pv


def solve_g(bp, disc, net, target):
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
        flows['fcfT'] = oT + cT
        flows['niT'] = nT
    fps = {}
    for c in COMPS:
        v = flows.get(c)
        if v is None or v <= 0:
            v = flows.get(FALLBACK.get(c, c))
        fps[c] = (v / sh) if (v is not None and v > 0) else 0.0
    ocfC = None
    so = sec['ocf']
    if len(so) >= 6 and so[5] and so[5] > 0 and so[0] and so[0] > 0:
        ocfC = ((so[0] / so[5]) ** 0.2 - 1) * 100
    fcfC = None
    if len(fcfs) >= 6 and fcfs[5] and fcfs[5] > 0 and fcfs[0] and fcfs[0] > 0:
        fcfC = ((fcfs[0] / fcfs[5]) ** 0.2 - 1) * 100
    rev = sec.get('revenue') or []
    marg = (ni / rev[0] * 100) if (rev and rev[0] and ni is not None) else 0.0
    co = (capex / ocf * 100) if (ocf and ocf > 0) else 0.0
    rev5C = 0.0
    if len(rev) >= 6 and rev[5] and rev[5] > 0 and rev[0] and rev[0] > 0:
        rev5C = ((rev[0] / rev[5]) ** 0.2 - 1) * 100
    feats = [g5, sg(g5), eny, sg(eny), ety, sg(ety), ocfC or 0.0, fcfC or 0.0,
             marg, sg(marg), co, sg(co), rev5C, sg(rev5C), 1.0]
    sd = [1.0 if d['sector'] == s2 else 0.0 for s2 in SECTORS]
    T.append(dict(t=t, sector=d['sector'], tgt=tgt, disc=disc, net=net,
                  fps=fps, x=np.array(feats + sd), sog=d['so_growth'], g5=g5))

N = len(T)
NF = len(T[0]['x'])
print(f'{N} tickers, {NF} features')


def base_ps(tk, alpha):
    a = alpha[tk['sector']]
    return sum(a[i] * tk['fps'][c] for i, c in enumerate(COMPS))


def g_interval(tk, bp, tol):
    lo = solve_g(bp, tk['disc'], tk['net'], tk['tgt'] * (1 - tol / 100))
    hi = solve_g(bp, tk['disc'], tk['net'], tk['tgt'] * (1 + tol / 100))
    if lo is None or hi is None: return None
    return (lo, hi)


def fit_growth_lp(ivls, hard_idx=None, center_pull=0.0):
    """min sum slack (+ center_pull * |w.x - center|) with optional hard set."""
    idx = [i for i, iv in enumerate(ivls) if iv is not None]
    ns = len(idx)
    nv = 2 * NF + ns + (ns if center_pull > 0 else 0)
    cvec = np.zeros(nv); cvec[2 * NF:2 * NF + ns] = 1.0
    if center_pull > 0:
        cvec[2 * NF + ns:] = center_pull
    rows_ub, b_ub = [], []
    hard = set(hard_idx or [])
    for k, i in enumerate(idx):
        lo, hi = ivls[i]; x = T[i]['x']
        sslack = 0.0 if i in hard else 1.0
        r = np.zeros(nv); r[:NF] = x; r[NF:2 * NF] = -x; r[2 * NF + k] = -sslack
        rows_ub.append(r); b_ub.append(hi)
        r = np.zeros(nv); r[:NF] = -x; r[NF:2 * NF] = x; r[2 * NF + k] = -sslack
        rows_ub.append(r); b_ub.append(-lo)
        if center_pull > 0:
            c = (lo + hi) / 2
            r = np.zeros(nv); r[:NF] = x; r[NF:2 * NF] = -x; r[2 * NF + ns + k] = -1
            rows_ub.append(r); b_ub.append(c)
            r = np.zeros(nv); r[:NF] = -x; r[NF:2 * NF] = x; r[2 * NF + ns + k] = -1
            rows_ub.append(r); b_ub.append(-c)
    res = linprog(cvec, A_ub=np.array(rows_ub), b_ub=np.array(b_ub),
                  bounds=[(0, None)] * nv, method='highs')
    if not res.success: return None
    return res.x[:NF] - res.x[NF:2 * NF]


def fit_alpha_lp(w, tol):
    alpha = {}
    for s in SECTORS:
        sk = [tk for tk in T if tk['sector'] == s]
        nc, ns = len(COMPS), len(sk)
        nv = nc + ns
        cvec = np.zeros(nv); cvec[nc:] = 1.0
        A_ub, b_ub = [], []
        for k, tk in enumerate(sk):
            g = max(-12.0, min(80.0, float(np.dot(w, tk['x']))))
            pc = pv_coef(g, tk['disc'])
            blo = (tk['tgt'] * (1 - tol / 100) - tk['net']) / pc
            bhi = (tk['tgt'] * (1 + tol / 100) - tk['net']) / pc
            f = np.array([tk['fps'][c] for c in COMPS])
            scale = max(abs(bhi), 1e-6)
            r = np.zeros(nv); r[:nc] = f; r[nc + k] = -scale
            A_ub.append(r); b_ub.append(bhi)
            r = np.zeros(nv); r[:nc] = -f; r[nc + k] = -scale
            A_ub.append(r); b_ub.append(-blo)
        A_eq = np.zeros((1, nv)); A_eq[0, :nc] = 1.0
        res = linprog(cvec, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * nv,
                      method='highs')
        if res.success:
            alpha[s] = res.x[:nc]
        else:
            v = np.zeros(nc); v[COMPS.index('ni')] = 1.0
            alpha[s] = v
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


def seed_alpha(kind):
    a = {}
    for s in SECTORS:
        v = np.zeros(len(COMPS))
        if kind in COMPS: v[COMPS.index(kind)] = 1
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
        else: v[COMPS.index('ni')] = 1
        a[s] = v
    return a


gbest = None  # (hits7, hits5, w, alpha, errs)
for kind in ['v9', 'mix', 'ni', 'ocf', 'fcf', 'fwdni', 'ocfT', 'fcfT', 'niT', 'ocf3']:
    alpha = seed_alpha(kind)
    w = None
    kbest = None
    for it in range(16):
        ivls = [g_interval(tk, base_ps(tk, alpha), TOL_FIT) for tk in T]
        w2 = fit_growth_lp(ivls)
        if w2 is None: break
        w = w2
        h, e = evaluate(w, alpha)
        h5 = sum(1 for v in e.values() if abs(v) <= 5)
        if kbest is None or (h, h5) > (kbest[0], kbest[1]):
            kbest = (h, h5, w.copy(), {s: alpha[s].copy() for s in alpha}, e)
        alpha = fit_alpha_lp(w, TOL_FIT)
        h, e = evaluate(w, alpha)
        h5 = sum(1 for v in e.values() if abs(v) <= 5)
        if (h, h5) > (kbest[0], kbest[1]):
            kbest = (h, h5, w.copy(), {s: alpha[s].copy() for s in alpha}, e)
    if kbest is None: continue
    print(f'seed={kind:5s}  best-iterate: {kbest[0]}/{N} ±7%, {kbest[1]}/{N} ±5%')
    if gbest is None or (kbest[0], kbest[1]) > (gbest[0], gbest[1]):
        gbest = kbest

h7, h5, w, alpha, errs = gbest
print(f'\n=== after alternation: {h7}/{N} ±7%, {h5}/{N} ±5% ===')

# ---- refinement: hard-lock current hits, center-pull, retry misses ----
for rounds in range(6):
    ivls = [g_interval(tk, base_ps(tk, alpha), TOL_FIT) for tk in T]
    hard = [i for i, tk in enumerate(T) if abs(errs[tk['t']]) <= TOL and ivls[i] is not None]
    w2 = fit_growth_lp(ivls, hard_idx=hard, center_pull=0.02)
    if w2 is not None:
        h2, e2 = evaluate(w2, alpha)
        h52 = sum(1 for v in e2.values() if abs(v) <= 5)
        if (h2, h52) > (h7, h5):
            w, errs, h7, h5 = w2, e2, h2, h52
    a2 = fit_alpha_lp(w, TOL_FIT)
    h2, e2 = evaluate(w, a2)
    h52 = sum(1 for v in e2.values() if abs(v) <= 5)
    if (h2, h52) > (h7, h5):
        alpha, errs, h7, h5 = a2, e2, h2, h52

print(f'=== after refinement: {h7}/{N} ±7%, {h5}/{N} ±5% ===\n')
print('per-ticker errors (sorted by |err|):')
for t, e in sorted(errs.items(), key=lambda kv: -abs(kv[1])):
    sec = next(tk['sector'] for tk in T if tk['t'] == t)
    mark = '' if abs(e) <= 5 else (' *' if abs(e) <= 7 else ' **MISS**')
    print(f'  {t:6s} {sec:11s} {e:+7.2f}%{mark}')

print('\nsector alphas (>1%):')
for s in SECTORS:
    aw = {c: round(float(v), 3) for c, v in zip(COMPS, alpha[s]) if v > 0.01}
    print(f'  {s:11s} {aw}')

names = ['g5', 'sqrt_g5', 'eny', 'sqrt_eny', 'ety', 'sqrt_ety', 'ocfC', 'fcfC',
         'marg', 'sqrt_marg', 'co', 'sqrt_co', 'rev5C', 'sqrt_rev5C', 'const'] + ['sec_' + s for s in SECTORS]
print('\ngrowth weights:')
for n, v in zip(names, w):
    print(f'  {n:14s} {v:+9.4f}')

with open('fit_v12.json', 'w') as f:
    json.dump({'weights': list(map(float, w)), 'feature_names': names,
               'sectors': SECTORS, 'components': COMPS,
               'alphas': {s: list(map(float, alpha[s])) for s in SECTORS},
               'hits7': h7, 'hits5': h5, 'n': N,
               'errors': {t: round(e, 3) for t, e in errs.items()}}, f, indent=1)
print('\nsaved fit_v12.json')
