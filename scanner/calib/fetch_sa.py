#!/usr/bin/env python3
"""Scrape stockanalysis.com TTM + annual flows for the calibration set -> sa_inputs.json.
Per ticker: ocfT, capexT, niT, epsT (TTM idx0) plus annual series (newest-first, excl TTM).
"""
import json, sys, time
sys.path.insert(0, '/home/user/webapp/scanner')
from vmi.stockanalysis import fetch_statement

inp = json.load(open('/home/user/webapp/scanner/calib/inputs2.json'))
out = {}
for t in inp:
    try:
        cf = fetch_statement(t, 'cashflow')
        ic = fetch_statement(t, 'income')
        dk = cf['datekey']
        ocf = cf['ncfo']; capex = cf['capex']; ni = cf['cash_flow_statement_net_income']
        eps = ic.get('epsDiluted') or ic.get('epsBasic')
        dki = ic['datekey']
        rec = {
            'datekey': dk[:9],
            'ocf': ocf[:9],
            'capex': capex[:9],
            'ni': ni[:9],
            'eps_datekey': dki[:9],
            'eps': eps[:9] if eps else None,
            'ttm0': dk[0] == 'TTM',
        }
        out[t] = rec
        print(t, 'OK', 'ttm0' if rec['ttm0'] else 'no-ttm', flush=True)
    except Exception as e:
        print(t, 'FAIL', repr(e)[:120], flush=True)
    time.sleep(0.4)

json.dump(out, open('/home/user/webapp/scanner/calib/sa_inputs.json', 'w'), indent=1)
print('saved', len(out), 'tickers')
