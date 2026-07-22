import time, numpy as np, pandas as pd
import pmcc_daily as P

books = P.load_books3()
ctx = P.prepare()
rf_daily = ctx[5]

def trim(arr, cut):
    dates = arr['dates']
    n = int(np.searchsorted(np.asarray(dates.values, dtype='datetime64[ns]'),
                            np.datetime64(cut)))
    a2 = dict(arr)
    for k, v in arr.items():
        try:
            if len(v) == len(dates):
                a2[k] = v[:n]
        except TypeError:
            pass
    return a2

for yr, cut in [(1990, '1993-12-31'), (2015, None)]:
    arr = P.arrays_for(yr, P.VINTAGES[yr], *ctx)
    if cut:
        arr = trim(arr, cut)
    rfa = rf_daily.loc[P.VINTAGES[yr]:].fillna(P.RF[yr])
    book = books[yr]['dow']
    for name, style, conv, ft in [('ds1', 'ds1', False, False),
                                  ('ds1_convert', 'ds1', True, False),
                                  ('hammer', 'hammer', False, False),
                                  ('ds1_full', 'ds1', False, True)]:
        t0 = time.time()
        ser, meta = P.run_pmcc(arr, book, P.RF[yr], style, conv, ft)
        c, dd, sp = P.stats_of(ser, rfa)
        print(f'{yr} {name:12s} cagr={c:6.2f}% dd={dd:6.1f} sh={sp:.3f} '
              f'final=${ser.iloc[-1]:,.0f} rolls={meta["long_rolls"]} '
              f'conv={meta["converts"]} cut={meta["cutlosses"]} '
              f'({time.time()-t0:.1f}s)', flush=True)
