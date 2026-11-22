"""Whole experiment suite for ONE data mode (run from that mode's src/ folder). Uses every core."""
import sys, time, pandas as pd, numpy as np
from runner import *
from policies import FULL_NAME, SIC_NAME
if __name__ == '__main__':
    t0 = time.time()
    A0, A2, A3, A6 = (POL[k] for k in ('A0 Incumbent rules', 'A2 +Deterministic MPC', 'A3 +Stochastic CVaR plan', FULL_NAME))
    jobs = []
    for name, pol in POL.items():                                    # 1) improvement ladder (full year)
        for s in SHORT: jobs.append(dict(tag=name, cluster=s, pol=pol))
    for s in SHORT: jobs.append(dict(tag=SIC_NAME, cluster=s, pol=A6, hw='sic'))
    for sg in (0.0, 0.25, 0.40, 0.55, 0.70, 0.85, 1.0):                # 2) steering effectiveness (max redirect share)
        for s in SHORT: jobs.append(dict(tag=f'sigma_{sg}', cluster=s, pol=A6, hw='sic', kw=dict(sigma=sg)))
    SC = {'base': (None, None), 'heatwave +5C & IEX x1.35 (14d)': ('heat', (130, 14, 5.0, 1.35)),
          'wind lull + cloud (7d, RE x0.3)': ('lull', (130, 7, 0.3)), 'IEX price spike x1.8 (10d)': ('price', (130, 10, 1.8)),
          'DISCOM feeder cut -50% 17-23h (10d)': ('outage', (130, 10, (0.5, 17, 23))), 'demand surge +30% (25d)': ('surge', (125, 25, 1.3)),
          'compound: heat+lull+spike x1.8+cut (10d)': ('compound', (130, 10, None))}
    pols = {'A0 Incumbent': (A0, 'std'), 'A2 Det. MPC': (A2, 'std'), 'A3 Stoch. MPC': (A3, 'std'), 'A7 Full AI+SiC': (A6, 'sic')}
    for sc, (mod, arg) in SC.items():                                  # 3) stress scenarios (25-day window)
        for pn, (pol, hw) in pols.items():
            for s in SHORT:
                jobs.append(dict(tag=f'SC|{sc}|{pn}', cluster=s, pol=pol, hw=hw, mod=mod, mod_arg=arg, kw=dict(day0=125, ndays=25)))
    SENS = {'elasticity x0.7': dict(kw=dict(eps_scale=0.7)), 'elasticity x1.3': dict(kw=dict(eps_scale=1.3)),
            'forecast error x2': dict(fc_noise=2.0), 'forecast error x3': dict(fc_noise=3.0),
            'IEX price level -25%': dict(mod='pricelevel', mod_arg=0.75), 'IEX price level +25%': dict(mod='pricelevel', mod_arg=1.25),
            'grid connection -15%': dict(kw=dict(grid_scale=0.85)), 'pack rebalancing effectiveness 0.3 (base 0.6)': dict(kw=dict(beta=0.3))}
    both = {'IEX price level -25%', 'IEX price level +25%', 'grid connection -15%'}   # baseline only changes for these
    for sn, extra in SENS.items():                                      # 4) one-at-a-time sensitivities (full year)
        for pn, (pol, hw) in (('A0 Incumbent', (A0, 'std')), ('A7 Full AI+SiC', (A6, 'sic'))):
            if pn == 'A0 Incumbent' and sn not in both: continue
            for s in SHORT:
                j = dict(tag=f'SE|{sn}|{pn}', cluster=s, pol=pol, hw=hw); j.update(extra); jobs.append(j)
    print(len(jobs), 'jobs on', NPROC, 'cores', flush=True)
    rows = run_many(jobs)
    pd.DataFrame(rows).to_csv('../out/tables/all_runs_raw.csv', index=False)
    print('done in', round(time.time() - t0), 's')
