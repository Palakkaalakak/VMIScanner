#!/usr/bin/env python3
"""For each ticker & base-flow candidate, solve the implied DCF20 growth g*
that reproduces the Base IV exactly (verified structure + CAPM).
Dump to implied_g2.json and print vs SOg / finviz g5 with sector tags."""
import json, math

RF, MRP = 3.608, 2.728
inp = json.load(open('inputs2.json'))

def dcf(base_ps, g_pct, disc_pct, debt_ps, cash_ps):
    g, r = g_pct/100.0, disc_pct/100.0
    f, pv = base_ps, 0.0
    for yr in range(1, 21):
        f *= 1 + (g if yr <= 10 else 0.04)
        pv += f/(1+r)**yr
    return pv - debt_ps + cash_ps

def implied_g(base_ps, disc, debt_ps, cash_ps, target):
    lo, hi = -50.0, 200.0
    if dcf(base_ps, hi, disc, debt_ps, cash_ps) < target: return None
    if dcf(base_ps, lo, disc, debt_ps, cash_ps) > target: return None
    for _ in range(80):
        mid = (lo+hi)/2
        if dcf(base_ps, mid, disc, debt_ps, cash_ps) < target: lo = mid
        else: hi = mid
    return (lo+hi)/2

out = {}
print(f"{'tkr':7s}{'sector':11s}{'SOg':>6s}{'g5':>7s} | implied g per base")
for t, d in sorted(inp.items()):
    fv, sec = d.get('finviz'), d.get('sec')
    if not fv or not sec: continue
    sh = fv.get('shares_outstanding'); beta = fv.get('beta') or 1.0
    if not sh: continue
    disc = RF + beta*MRP
    tgt = d['target_iv']
    ocf = sec['ocf'][0] if sec['ocf'] else None
    capex = abs(sec['capex'][0]) if sec['capex'] and sec['capex'][0] is not None else None
    ni = sec['netIncome'][0] if sec['netIncome'] else None
    fcf = (ocf-capex) if (ocf is not None and capex is not None) else None
    fwd = fv.get('fwd_eps')
    debt = ((sec.get('std') or 0)+(sec.get('ltd') or 0))/sh
    cash = ((sec.get('cash') or 0)+(sec.get('sti') or 0))/sh
    rec = {'sector': d['sector'], 'so_growth': d['so_growth'],
           'g5': fv.get('eps_next_5y'), 'disc': disc, 'impl': {}}
    parts = []
    for name, flow in (('ocf',ocf),('fcf',fcf),('ni',ni),('fwdni', fwd*sh if fwd else None)):
        if flow is None or flow <= 0: continue
        ig = implied_g(flow/sh, disc, debt, cash, tgt)
        if ig is not None:
            rec['impl'][name] = round(ig, 3)
            parts.append(f"{name}:{ig:6.2f}")
    out[t] = rec
    g5 = fv.get('eps_next_5y')
    g5s = f"{g5:6.2f}" if g5 is not None else "    --"
    print(f"{t:7s}{d['sector']:11s}{d['so_growth']:6.1f}{g5s} | " + '  '.join(parts))

json.dump(out, open('implied_g2.json','w'), indent=1)
print('\nwrote implied_g2.json')
