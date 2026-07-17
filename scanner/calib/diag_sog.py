#!/usr/bin/env python3
"""Diagnostic: plug SO's own growth column into verified DCF20 structure.
For each ticker, compute IV per base-flow candidate; report best base + err.
Flags tickers where NO base gets close (currency issues / different model)."""
import json, math

RF, MRP = 3.608, 2.728

inp = json.load(open('inputs2.json'))

def cagr6(series):
    # newest-first list; 6-point CAGR in %
    s = [x for x in series if x is not None]
    if len(s) < 6: return None
    new, old = s[0], s[5]
    if old is None or new is None or old <= 0 or new <= 0: return None
    return ((new/old)**(1/5)-1)*100

def dcf(base_ps, g_pct, disc_pct, debt_ps, cash_ps):
    g, r = g_pct/100.0, disc_pct/100.0
    f, pv = base_ps, 0.0
    for yr in range(1, 21):
        f *= 1 + (g if yr <= 10 else 0.04)
        pv += f / (1+r)**yr
    return pv - debt_ps + cash_ps

rows = []
for t, d in sorted(inp.items()):
    fv, sec = d.get('finviz'), d.get('sec')
    if not fv or not sec: 
        rows.append((t, d['sector'], None, None, None, 'no-data')); continue
    sh = fv.get('shares_outstanding')
    beta = fv.get('beta') or 1.0
    tgt = d['target_iv']; sog = d['so_growth']
    disc = RF + beta*MRP
    ocf = sec['ocf'][0] if sec['ocf'] else None
    capex = sec['capex'][0] if sec['capex'] else None
    ni = sec['netIncome'][0] if sec['netIncome'] else None
    fcf = (ocf - abs(capex)) if (ocf is not None and capex is not None) else None
    debt = (sec.get('std') or 0) + (sec.get('ltd') or 0)
    cash = (sec.get('cash') or 0) + (sec.get('sti') or 0)
    cands = {'ocf': ocf, 'fcf': fcf, 'ni': ni}
    fwd = fv.get('fwd_eps')
    if fwd: cands['fwdni'] = fwd * sh
    best = None
    errs = {}
    for name, flow in cands.items():
        if flow is None or flow <= 0 or not sh: continue
        iv = dcf(flow/sh, sog, disc, debt/sh, cash/sh)
        e = (iv/tgt - 1)*100
        errs[name] = e
        if best is None or abs(e) < abs(best[1]): best = (name, e)
    rows.append((t, d['sector'], best, errs, sog, ''))

print(f"{'tkr':7s}{'sector':11s}{'SOg':>6s}  best-base  err%    all-errs")
bad = []
for t, sec_, best, errs, sog, note in rows:
    if best is None:
        print(f"{t:7s}{sec_:11s}   --   {note}"); continue
    es = ' '.join(f"{k}:{v:+.1f}" for k,v in errs.items())
    flag = ' <<<' if abs(best[1]) > 7 else ''
    if abs(best[1]) > 7: bad.append(t)
    print(f"{t:7s}{sec_:11s}{sog:6.1f}  {best[0]:8s}{best[1]:+7.2f}  {es}{flag}")
print('\nTickers with NO base within ±7% using SOg directly:', bad)
