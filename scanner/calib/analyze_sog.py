#!/usr/bin/env python3
"""What is SO's 'Avg. Growth Rates' column? Test candidate compositions:
finviz g5, historical CAGRs (rev/ni/ocf/fcf), and combinations.
Also: per-sector bias of SOg vs finviz g5, and SOg vs implied-g gaps."""
import json, math, statistics as st
from collections import defaultdict

inp = json.load(open('inputs2.json'))
impl = json.load(open('implied_g2.json'))

def cagrN(series, n):
    s = series or []
    if len(s) < n+1: return None
    new, old = s[0], s[n]
    if new is None or old is None or old <= 0 or new <= 0: return None
    return ((new/old)**(1/n)-1)*100

rows = []
for t, d in sorted(inp.items()):
    fv, sec = d.get('finviz'), d.get('sec')
    if not fv or not sec: continue
    sog = d['so_growth']; g5 = fv.get('eps_next_5y')
    ep5 = fv.get('eps_past_5y')
    fcfs = [(o-abs(c)) if (o is not None and c is not None) else None
            for o,c in zip(sec['ocf'], sec['capex'])]
    feats = {
        'g5': g5, 'eps_past_5y': ep5,
        'revC5': cagrN(sec['revenue'],5), 'niC5': cagrN(sec['netIncome'],5),
        'ocfC5': cagrN(sec['ocf'],5), 'fcfC5': cagrN(fcfs,5),
        'revC3': cagrN(sec['revenue'],3), 'niC3': cagrN(sec['netIncome'],3),
        'ocfC3': cagrN(sec['ocf'],3), 'fcfC3': cagrN(fcfs,3),
    }
    rows.append((t, d['sector'], sog, feats))

# 1) single-feature errors vs SOg
print('=== |SOg - feature| stats (n = tickers where feature avail) ===')
for f in ['g5','eps_past_5y','revC5','niC5','ocfC5','fcfC5','revC3','niC3','ocfC3','fcfC3']:
    ds = [(sog - feats[f]) for _,_,sog,feats in rows if feats[f] is not None]
    if len(ds) < 5: continue
    print(f"{f:13s} n={len(ds):2d} mean={st.mean(ds):+6.2f} med={st.median(ds):+6.2f} mad={st.median(sorted(abs(x) for x in ds)):5.2f}")

# 2) simple averages of subsets
import itertools
print('\n=== SOg vs averages of feature subsets (med abs dev) ===')
cand = ['g5','revC5','niC5','ocfC5','fcfC5','eps_past_5y']
best = []
for r in range(1, 5):
    for combo in itertools.combinations(cand, r):
        ds = []
        for _,_,sog,feats in rows:
            vals = [feats[f] for f in combo if feats[f] is not None]
            if len(vals) != len(combo): continue
            ds.append(sog - sum(vals)/len(vals))
        if len(ds) < 25: continue
        mad = st.median(sorted(abs(x) for x in ds))
        best.append((mad, combo, len(ds), st.mean(ds)))
best.sort()
for mad, combo, n, mn in best[:12]:
    print(f"mad={mad:5.2f} mean={mn:+5.2f} n={n:2d}  {'+'.join(combo)}")

# 3) per-sector SOg - g5 bias
print('\n=== per-sector (SOg - finviz g5) ===')
by = defaultdict(list)
for t, s, sog, feats in rows:
    if feats['g5'] is not None: by[s].append((t, sog - feats['g5']))
for s, xs in sorted(by.items()):
    vs = [v for _,v in xs]
    print(f"{s:11s} n={len(vs):2d} mean={st.mean(vs):+6.2f} med={st.median(vs):+6.2f}  " +
          ' '.join(f"{t}:{v:+.1f}" for t,v in xs))

# 4) per-sector gap: implied g (per base) minus SOg  -> which base does SO use per sector?
print('\n=== per-sector median (implied_g[base] - SOg) ===')
for base in ('ocf','fcf','ni','fwdni'):
    by2 = defaultdict(list)
    for t, r in impl.items():
        ig = r['impl'].get(base)
        if ig is not None: by2[r['sector']].append(ig - r['so_growth'])
    line = f"{base:6s} "
    for s in sorted(by2):
        line += f" {s}:{st.median(by2[s]):+5.1f}(n{len(by2[s])})"
    print(line)
