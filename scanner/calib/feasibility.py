#!/usr/bin/env python3
"""Feasibility map: per ticker, ±7% growth intervals per base; check which
candidate predictors (g5, SOg, shipped blend) fall inside ANY base interval.
This tells us the theoretical ceiling before distilling a base rule."""
import json, math
import numpy as np

RF, MRP = 3.608, 2.728
TOL = 7.0
inp = json.load(open('inputs2.json'))
P = json.load(open('blend_params_sec.json'))
W = P['weights']

def cagrN(series, n):
    s = series or []
    if len(s) < n+1: return None
    new, old = s[0], s[n]
    if new is None or old is None or old <= 0 or new <= 0: return None
    return ((new/old)**(1.0/n)-1)*100

def dcf(base_ps, g_pct, disc_pct, debt_ps, cash_ps):
    g, r = g_pct/100.0, disc_pct/100.0
    f, pv = base_ps, 0.0
    for yr in range(1,21):
        f *= 1+(g if yr<=10 else 0.04)
        pv += f/(1+r)**yr
    return pv - debt_ps + cash_ps

def solve_g(bp, disc, debt, cash, target):
    lo, hi = -60.0, 300.0
    if dcf(bp,hi,disc,debt,cash) < target: return None
    if dcf(bp,lo,disc,debt,cash) > target: return None
    for _ in range(80):
        mid=(lo+hi)/2
        if dcf(bp,mid,disc,debt,cash) < target: lo=mid
        else: hi=mid
    return (lo+hi)/2

def sg(x): return math.copysign(math.sqrt(abs(x)), x)

def shipped_blend(g5, eny, ety, ocfC, fcfC):
    feats=[g5, sg(g5), eny, sg(eny), ety, sg(ety), ocfC, fcfC, 1.0]
    return sum(w*f for w,f in zip(W,feats))

rows=[]
for t,d in sorted(inp.items()):
    fv,sec = d.get('finviz'), d.get('sec')
    if not fv or not sec or t in ('NVO','BABA'): continue
    sh=fv.get('shares_outstanding'); g5=fv.get('eps_next_5y')
    eny,ety=fv.get('eps_next_y'),fv.get('eps_this_y')
    if not sh or g5 is None: continue
    beta=fv.get('beta') or 1.0; disc=RF+beta*MRP; tgt=d['target_iv']
    ocf=sec['ocf'][0] if sec['ocf'] else None
    capex=abs(sec['capex'][0]) if sec['capex'] and sec['capex'][0] is not None else None
    ni=sec['netIncome'][0] if sec['netIncome'] else None
    fcf=(ocf-capex) if (ocf is not None and capex is not None) else None
    fwd=fv.get('fwd_eps'); fwdni=fwd*sh if fwd else None
    debt=((sec.get('std') or 0)+(sec.get('ltd') or 0))/sh
    cash=((sec.get('cash') or 0)+(sec.get('sti') or 0))/sh
    fcfs=[(o-abs(c)) if (o is not None and c is not None) else None for o,c in zip(sec['ocf'],sec['capex'])]
    ocfC=cagrN(sec['ocf'],5) or 0.0; fcfC=cagrN(fcfs,5) or 0.0
    ivls={}
    for name,flow in (('ocf',ocf),('fcf',fcf),('ni',ni),('fwdni',fwdni)):
        if flow is None or flow<=0: continue
        bp=flow/sh
        lo=solve_g(bp,disc,debt,cash,tgt*(1-TOL/100))
        hi=solve_g(bp,disc,debt,cash,tgt*(1+TOL/100))
        if lo is None or hi is None: continue
        ivls[name]=(lo,hi)
    sb=shipped_blend(g5,eny,ety,ocfC,fcfC) if (eny is not None and ety is not None) else None
    rows.append((t,d['sector'],d['so_growth'],g5,sb,ivls))

def inside(v,ivls):
    return [b for b,(lo,hi) in ivls.items() if lo<=v<=hi] if v is not None else []

cnt={'g5':0,'sog':0,'blend':0,'any_union':0}
print(f"{'tkr':7s}{'sector':11s}{'SOg':>6s}{'g5':>7s}{'blend':>7s} | ±7% intervals -> hits")
for t,s,sog,g5,sb,ivls in rows:
    iv_s=' '.join(f"{b}[{lo:.1f},{hi:.1f}]" for b,(lo,hi) in ivls.items())
    h_g5, h_sog, h_b = inside(g5,ivls), inside(sog,ivls), inside(sb,ivls)
    if h_g5: cnt['g5']+=1
    if h_sog: cnt['sog']+=1
    if h_b: cnt['blend']+=1
    if h_g5 or h_sog or h_b: cnt['any_union']+=1
    sbs=f"{sb:7.1f}" if sb is not None else "     --"
    print(f"{t:7s}{s:11s}{sog:6.1f}{g5:7.1f}{sbs} | {iv_s}")
    print(f"        hits: g5->{h_g5}  SOg->{h_sog}  blend->{h_b}")
n=len(rows)
print(f"\ncoverage (pred inside SOME base's ±7% interval): g5 {cnt['g5']}/{n}, SOg {cnt['sog']}/{n}, shipped-blend {cnt['blend']}/{n}")
