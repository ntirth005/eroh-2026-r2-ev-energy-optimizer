import json, pickle, os, numpy as np, pandas as pd, time
from multiprocessing import Pool
import datagen as dg, engine as E, lp as L
from params import *
from policies import POL, HW_SIC

_cal = json.load(open('../data/calib.json'))
_fc = pickle.load(open('../data/forecasts.pkl', 'rb'))
_cl, _ = dg.make_base()
_FR = dg.apply_calib(_cl, _cal)
NPROC = os.cpu_count() or 8


def _i0(df):
    return len(df) - 8760


def job(a):
    """a = dict(tag, cluster, pol, hw('std'|'sic'), kw, mod, mod_arg, fc_noise, cal_mod)"""
    s = a['cluster']
    df = _FR[s]
    if a.get('mod'):
        df = MODS[a['mod']](df.copy(), s, a.get('mod_arg'))
    hw = HW_SIC if a.get('hw') == 'sic' else L.HW()
    cal = _cal
    fc = _fc[s]
    if a.get('fc_noise'):
        # make the forecasts k times worse in total: extra log-noise sd = base_sd * sqrt(k^2 - 1), on demand, RE AND price
        k = a['fc_noise']; extra = np.sqrt(max(k * k - 1.0, 0.0))
        fc = dict(fc); rng = np.random.default_rng(5); i0 = _i0(df)
        for kk, sd in (('dc', 0.11), ('sw', 0.11), ('dam', 0.15)):
            v = fc[kk].copy(); v[i0:] = v[i0:] * np.exp(rng.normal(0, sd * extra, len(v) - i0)); fc[kk] = v
        v = fc['re'].copy(); v[i0:] = np.maximum(v[i0:] * np.exp(rng.normal(0, 0.20 * extra, len(v) - i0)), 0); fc['re'] = v
    t = time.time()
    R = E.simulate(s, df, cal, a['pol'], hw=hw, fc=fc, eps_hat=_fc[s]['eps_hat'], **a.get('kw', {}))
    k = E.kpis(R, cal[s]['fixed_cr'])
    k.update(cluster=s, tag=a['tag'], secs=time.time() - t)
    os.makedirs('../data/runs', exist_ok=True)
    safe = ''.join(ch if ch.isalnum() else '_' for ch in a['tag'])
    R.to_pickle(f"../data/runs/{safe}__{s}.pkl")
    return k


MODS = {}


def _win(df, d0, nd):
    i0 = _i0(df) + d0 * 24
    return i0, i0 + nd * 24


def _set(df, col, a, b, vals):
    df.iloc[a:b, df.columns.get_loc(col)] = vals


def mod_heat(df, s, arg):                      # heatwave: +dT degC for nd days (pf = accompanying IEX price factor)
    d0, nd, dT, pf = arg
    a, b = _win(df, d0, nd)
    T0 = df['T'].values[a:b].copy(); T1 = T0 + dT
    _set(df, 'T', a, b, T1)
    _set(df, 'dc_ref', a, b, df['dc_ref'].values[a:b] * (1 + 0.016 * np.maximum(T1 - 30, 0)) / (1 + 0.016 * np.maximum(T0 - 30, 0)))
    _set(df, 'sw_ref', a, b, df['sw_ref'].values[a:b] * (1 + 0.010 * np.maximum(T1 - 30, 0)) / (1 + 0.010 * np.maximum(T0 - 30, 0)))
    _set(df, 're', a, b, df['re'].values[a:b] * (1 - 0.004 * dT * 0.5))
    _set(df, 'dam', a, b, np.minimum(df['dam'].values[a:b] * pf, PRICE_CAP_NOW))
    return df


def mod_lull(df, s, arg):                      # wind lull + heavy cloud: renewable output x k
    d0, nd, k = arg
    a, b = _win(df, d0, nd)
    _set(df, 're', a, b, df['re'].values[a:b] * k)
    return df


def mod_price(df, s, arg):                     # IEX price spike x k (capped at the current exchange cap)
    d0, nd, k = arg
    a, b = _win(df, d0, nd)
    _set(df, 'dam', a, b, np.minimum(df['dam'].values[a:b] * k, PRICE_CAP_NOW))
    return df


def mod_outage(df, s, arg):                    # DISCOM / feeder constraint: connection limit x k for hours h0..h1 on nd days
    d0, nd, (k, h0, h1) = arg
    a, b = _win(df, d0, nd)
    G = np.full(len(df), _cal[s]['grid'])
    for d in range(nd):
        G[a + d * 24 + h0: a + d * 24 + h1] = _cal[s]['grid'] * k
    df['G'] = G
    return df


def mod_surge(df, s, arg):                     # demand surge (festival season / fleet onboarding)
    d0, nd, k = arg
    a, b = _win(df, d0, nd)
    for c in ('dc_ref', 'sw_ref'):
        _set(df, c, a, b, df[c].values[a:b] * k)
    return df


def mod_compound(df, s, arg):                  # heat (+5 C, no price factor) + lull + price x1.8 + feeder cut, all at once
    d0, nd, _ = arg
    df = mod_heat(df, s, (d0, nd, 5.0, 1.0)); df = mod_lull(df, s, (d0, nd, 0.35)); df = mod_price(df, s, (d0, nd, 1.8))
    return mod_outage(df, s, (d0, nd, (0.7, 17, 23)))


def mod_pricelevel(df, s, arg):
    df['dam'] = df['dam'].values * arg
    return df


MODS.update(heat=mod_heat, lull=mod_lull, price=mod_price, outage=mod_outage, surge=mod_surge, compound=mod_compound, pricelevel=mod_pricelevel)


def run_many(jobs, procs=None):
    with Pool(procs or NPROC) as p:
        return p.map(job, jobs, chunksize=1)
