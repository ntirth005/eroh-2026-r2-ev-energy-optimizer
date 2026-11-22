"""Fit the semi-empirical battery ageing law from the state-of-health audit records and write data/deg.json.
fade = a*sqrt(age)*2^((T-25)/10)  +  b*EFC/100  +  b*c*EFC/100*max(T-25,0)"""
import json, numpy as np, pandas as pd
import datagen as dg
rec = dg.soh_records()
out = {}
rows = []
for kind in ('bess', 'pack'):
    r = rec[rec.kind == kind]
    y = 1 - r.soh.values
    x1 = np.sqrt(r.age_yr.values) * 2 ** ((r.temp.values - 25) / 10)
    x2 = r.efc.values / 100
    x3 = x2 * np.maximum(r.temp.values - 25, 0)
    X = np.column_stack([x1, x2, x3])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, bc = beta
    out[kind + '_cycles'] = float(100 * 0.2 / b)                 # equivalent full cycles to 20 % fade (80 % SoH) at 25 C
    out[kind + '_heat'] = float(bc / b)
    res = y - X @ beta
    rows.append((kind, a, b, bc / b, out[kind + '_cycles'], 1 - res.var() / y.var()))
out['heat_wear'] = float(np.mean([out['bess_heat'], out['pack_heat']]))
json.dump(out, open('../data/deg.json', 'w'), indent=1)
pd.DataFrame(rows, columns=['unit', 'calendar_coef', 'cycle_coef_per100EFC', 'heat_factor_per_degC', 'EFC_to_80pct_SoH', 'R2']).to_csv('../out/tables/degradation_fit.csv', index=False)
print(pd.DataFrame(rows, columns=['unit', 'cal', 'cyc/100EFC', 'heat/degC', 'EFC life', 'R2']).round(4).to_string()); print(out)
