"""Calibrate the BASELINE (incumbent operations) to the case's current KPIs: reliability ~78 %, losses ~10 %.
Physical: grid connection = 1.0 x p99 of test-year bus load.
Calibrated by bisection per cluster:
  sw_sd  - dispersion of swap demand across stations (drives stock-outs at hot stations)   -> swap fill rate target
  hub_sd - dispersion of DC demand across hubs (drives queues / lost sessions at hot hubs) -> cluster reliability target
  fixed  - fixed cost so the incumbent's cluster EBITDA margin equals the stated starting margin
The incumbent schedules COV = 92 % of its (hub-deliverable) naive net-load forecast."""
import json, numpy as np, pandas as pd
import datagen as dg, engine as E
from params import *

cl, _ = dg.make_base()
TARGET = dict(DEL=0.76, MUM=0.74, BLR=0.80, HYD=0.81, MAA=0.83)              # cluster reliability (assumed spread, ~78 % overall)
TARGET_SW = dict(DEL=0.82, MUM=0.80, BLR=0.88, HYD=0.92, MAA=0.93)           # swap fill rate (stock-outs concentrated in DEL / MUM)
MARGIN0 = dict(DEL=0.10, MUM=0.12, BLR=0.08, HYD=0.07, MAA=0.06)
COV = 0.92
cal = {s: dict(DEFAULT_CALIB[s]) for s in SHORT}
cal['DEL'].update(dc_mw=62.0, sw_mw=9.0); cal['MUM'].update(dc_mw=62.0, sw_mw=8.5); cal['BLR'].update(dc_mw=48.0, sw_mw=6.5)
cal['HYD'].update(dc_mw=34.0, sw_mw=4.0); cal['MAA'].update(dc_mw=24.0, sw_mw=3.0)
fr = dg.apply_calib(cl, cal)
for s in SHORT:
    d = fr[s]
    bus = (d.loc[dg.TESTM, 'dc_ref'] / ETA_DC + d.loc[dg.TESTM, 'sw_ref'] / ETA_SW) / (1 - DIST_LOSS)
    cal[s]['grid'] = round(float(np.percentile(bus, 99)), 1)
fr = dg.apply_calib(cl, cal)
BASE = dict(fc='naive', da='rule', price='tou', rt='rule', sell_re=False, pi=0.0, cov=COV)


def kp(s):
    return E.kpis(E.simulate(s, fr[s], cal, BASE))


for s in SHORT:
    cal[s]['hub_sd'] = 1.0
    lo, hi = 0.01, 2.5                                   # swap stations
    for _ in range(14):
        mid = 0.5 * (lo + hi); cal[s]['sw_sd'] = mid
        if kp(s)['rel_sw'] > TARGET_SW[s]: lo = mid      # more dispersion -> more stock-outs
        else: hi = mid
    cal[s]['sw_sd'] = round(lo, 3)
    lo, hi = 0.01, 2.5                                   # DC hubs
    for _ in range(14):
        mid = 0.5 * (lo + hi); cal[s]['hub_sd'] = mid
        if kp(s)['reliability'] > TARGET[s]: lo = mid
        else: hi = mid
    cal[s]['hub_sd'] = round(lo, 3)
    cal[s]['cov'] = COV
tot_rev = tot_e = 0.0
cal['_margin0'] = MARGIN0
for s in SHORT:
    k = kp(s)
    # EBITDA before fixed cost -> choose fixed cost so margin == MARGIN0
    R = E.simulate(s, fr[s], cal, BASE)
    k0 = E.kpis(R, 0.0)
    cal[s]['fixed_cr'] = round(k0['ebitda_cr'] - MARGIN0[s] * k0['revenue_cr'], 1)
    k2 = E.kpis(R, cal[s]['fixed_cr']); tot_rev += k2['revenue_cr']; tot_e += k2['ebitda_cr']
    print(s, 'G', cal[s]['grid'], 'hub_sd', cal[s]['hub_sd'], 'sw_sd', cal[s]['sw_sd'], 'fixed Cr', cal[s]['fixed_cr'],
          'Rs/kW-yr', round(cal[s]['fixed_cr'] * 1e7 / (CFG[s]['cap'] * 1000)),
          'rel %.3f (dc %.3f sw %.3f) loss %.3f margin %.3f re_util %.2f' % (k2['reliability'], k2['rel_dc'], k2['rel_sw'], k2['loss_pct'], k2['margin'], k2['re_util']))
print('network margin', round(tot_e / tot_rev, 4), 'revenue Cr', round(tot_rev))
json.dump(cal, open('../data/calib.json', 'w'), indent=1)
