#!/usr/bin/env python3
"""Run the SHIPPED calibrated model (base rule + growth blend) on the 42-ticker
set and report per-ticker + per-sector residuals."""
import json, math
from collections import defaultdict

RF, MRP = 3.608, 2.728
P = json.load(open('blend_params_sec.json'))
W = P['weights']; R = P['rule']
inp = json.load(open('inputs2.json'))

def cagr6(series):
    s = series or []
    if len(s) < 6: return None
    new, old = s[0], s[5]
    if new is None or old is None or old <= 0 or new <= 0: return None
    return ((new/old)**(1/5)-1)*100

def ssq(x):
    return math.copysign(math.sqrt(abs(x)), 1.0) if x >= 0 else -math.sqrt(abs(x))

def blend_g(g5, eny, ety, ocfC, fcfC):
    sq = lambda v: math.sqrt(abs(v))*(1 if v>=0 else -1)
    feats = [g5, math.sqrt(abs(g5))*(1 if g5>=0 else -1),
             eny, sq(eny), ety, sq(ety), ocfC, fcfC, 1.0]
    return sum(w*f for w,f in zip(W, feats))

def dcf(base_ps, g_pct, disc_pct, debt_ps, cash_ps):
    g, r = g_pct/100.0, disc_pct/100.0
    f, pv = base_ps, 0.0
    for yr in range(1, 21):
        f *= 1 + (g if yr <= 10 else 0.04)
        pv += f/(1+r)**yr
    return pv - debt_ps + cash_ps

def pick_base(capex_ocf, g5, ocf, fcf, ni):
    m = {'ocf': ocf, 'fcf': fcf, 'ni': ni}
    if capex_ocf is None:
        leaf = 'ni'
    elif capex_ocf <= R['a']: leaf = R['leaves'][0]
    elif g5 is not None and g5 <= R['tg']: leaf = R['leaves'][1]
    elif capex_ocf <= R['b']: leaf = R['leaves'][2]
    elif capex_ocf <= R['c']: leaf = R['leaves'][3]
    else: leaf = R['leaves'][4]
    v = m.get(leaf)
    if v is None or v <= 0:
        for cand in ('ocf','fcf','ni'):
            if m[cand] and m[cand] > 0: return cand, m[cand]
        return None, None
    return leaf, v

rows, by_sector = [], defaultdict(list)
for t, d in sorted(inp.items()):
    fv, sec = d.get('finviz'), d.get('sec')
    if not fv or not sec:
        rows.append((t, d['sector'], None, None, None, 'no-data')); continue
    sh = fv.get('shares_outstanding'); beta = fv.get('beta') or 1.0
    tgt = d['target_iv']
    g5 = fv.get('eps_next_5y'); eny = fv.get('eps_next_y'); ety = fv.get('eps_this_y')
    if g5 is None or eny is None or ety is None or not sh:
        rows.append((t, d['sector'], None, None, None, 'no-est')); continue
    ocf = sec['ocf'][0] if sec['ocf'] else None
    capex = abs(sec['capex'][0]) if sec['capex'] and sec['capex'][0] is not None else None
    ni = sec['netIncome'][0] if sec['netIncome'] else None
    fcf = (ocf-capex) if (ocf is not None and capex is not None) else None
    co = (capex/ocf*100) if (capex is not None and ocf and ocf > 0) else None
    ocfC = cagr6(sec['ocf']); fcfC = cagr6([ (o-abs(c)) if (o is not None and c is not None) else None for o,c in zip(sec['ocf'],sec['capex'])])
    if ocfC is None: ocfC = 0.0
    if fcfC is None: fcfC = 0.0
    g = blend_g(g5, eny, ety, ocfC, fcfC)
    leaf, flow = pick_base(co, g5, ocf, fcf, ni)
    if flow is None:
        rows.append((t, d['sector'], None, None, None, 'no-flow')); continue
    debt = (sec.get('std') or 0)+(sec.get('ltd') or 0)
    cash = (sec.get('cash') or 0)+(sec.get('sti') or 0)
    disc = RF + beta*MRP
    iv = dcf(flow/sh, g, disc, debt/sh, cash/sh)
    e = (iv/tgt-1)*100
    rows.append((t, d['sector'], e, g, leaf, ''))
    by_sector[d['sector']].append(e)

print(f"{'tkr':7s}{'sector':11s}{'err%':>8s}  {'g':>6s}  base  SOg")
for t, s, e, g, leaf, note in rows:
    if e is None: print(f"{t:7s}{s:11s}    --  {note}"); continue
    sog = inp[t]['so_growth']
    flag = ' <<<' if abs(e) > 7 else ''
    print(f"{t:7s}{s:11s}{e:+8.2f}  {g:6.1f}  {leaf:4s}  {sog:5.1f}{flag}")

print('\n--- per-sector residuals (shipped model) ---')
for s, es in sorted(by_sector.items()):
    import statistics as st
    print(f"{s:11s} n={len(es):2d} mean={st.mean(es):+7.2f} median={st.median(es):+7.2f} max|e|={max(abs(x) for x in es):6.2f}")
allE = [e for _,_,e,_,_,_ in rows if e is not None]
import statistics as st
print(f"\nALL n={len(allE)} mean={st.mean(allE):+.2f} med={st.median(allE):+.2f} max|e|={max(abs(x) for x in allE):.2f} within±7%: {sum(1 for e in allE if abs(e)<=7)}/{len(allE)}")
