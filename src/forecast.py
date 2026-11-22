"""ML forecasting layer: price-elasticity estimation, LightGBM forecasts (cluster demand, RE output, IEX DAM price),
hierarchical reconciliation (cluster -> network, WLS/MinT-diagonal) and the forecast-error scenario library used by
the stochastic optimiser.  Train Nov-2021..Nov-2025, test the last 365 days (strictly out-of-sample). Elasticity is estimated on training rows only;
demand is detrended (log-linear trend fitted on training years) so trees do not have to extrapolate growth."""
import json, pickle, time, numpy as np, pandas as pd, lightgbm as lgb
import datagen as dg
from params import *

cal = json.load(open('../data/calib.json'))
cl_base, _ = dg.make_base()
FR = dg.apply_calib(cl_base, cal)
IDX = dg.IDX
HOUR, DOW, MONTH, DOY, YEAR = dg.HOUR, dg.DOW, dg.MONTH, dg.DOY, dg.YEAR
N = len(IDX)
TR = YEAR <= 2025                       # training: 23 Nov 2021 - 22 Nov 2025
TE = YEAR == 2026                       # test: 23 Nov 2025 - 22 Nov 2026 (strictly out-of-sample)
BLOCKS = [(0, 6), (6, 12), (12, 18), (18, 24)]
dayoff = np.clip((DOW >= 5).astype(float) + dg.IS_HOL, 0, 1)


def block_of(h):
    return np.array([min(h_ // 6, 3) for h_ in h])


# ------------------------------------------------------------------------------------------------------------
def estimate_elasticity(df, col):
    """log-log regression with hour x day-type fixed effects, month FE, trend, heat; elasticity by 6-h block"""
    y = np.log(df[col].values)[TR]
    X = []
    for h in range(24):
        for d in (0, 1):
            X.append(((HOUR == h) & (dayoff == d)).astype(float))
    for m in range(2, 13):
        X.append((MONTH == m).astype(float))
    X.append((YEAR - 2023) + DOY / 365.25)
    X.append(np.maximum(df['T'].values - 30, 0))
    X.append((df['cloud'].values > 0.85).astype(float))
    lm = np.log(df['m_hist'].values)
    nfe = len(X)
    b = block_of(HOUR)
    for k in range(4):
        X.append(lm * (b == k))
    X = np.column_stack(X)[TR]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    eps_b = beta[nfe:nfe + 4]
    return eps_b


def lagfeat(y):
    s = pd.Series(y)
    f = {'lag48': s.shift(48).values, 'lag168': s.shift(168).values}
    m = np.mean([s.shift(24 * k).values for k in range(2, 9)], axis=0)
    f['wk_mean'] = m
    f['d_mean'] = s.shift(48).rolling(24).mean().values
    return f


def feats(df, y, kind, lat=None, nat=None):
    F = dict(hour=HOUR, dow=DOW, month=MONTH, doy=DOY, hol=dg.IS_HOL, dayoff=dayoff, t=(np.arange(N) / 8760.0))
    F.update(lagfeat(y))
    if kind in ('dc', 'sw'):
        F.update(T_fc=df['T_fc'].values, cl_fc=df['cloud_fc'].values, heat_fc=np.maximum(df['T_fc'].values - 30, 0))
    elif kind == 're':
        F.update(T_fc=df['T_fc'].values, cl_fc=df['cloud_fc'].values, w_fc=df['wind_fc'].values,
                 solcf_hint=np.clip(np.maximum(np.sin(np.deg2rad(15 * (HOUR + 0.5 - 12) + 90)), 0), 0, 1) * (1 - 0.78 * df['cloud_fc'].values ** 1.4))
    elif kind == 'dam':
        F.update(T_fc=nat['T'], cl_fc=nat['cl'], w_fc=nat['w'], heat_fc=np.maximum(nat['T'] - 31, 0),
                 m30=pd.Series(y).shift(48).rolling(24 * 30, min_periods=24).mean().values)
    return pd.DataFrame(F)


PARAMS = dict(n_estimators=350, learning_rate=0.05, num_leaves=48, min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=6)


PARAMS_PRICE = dict(PARAMS, n_estimators=250, num_leaves=15, min_child_samples=60, learning_rate=0.04)


def fit_predict(X, y, tr_mask, pr_mask, params=None):
    ok = tr_mask & np.isfinite(y) & np.isfinite(X['lag168'].values)
    m = lgb.LGBMRegressor(**(params or PARAMS))
    m.fit(X[ok], y[ok])
    return m.predict(X[pr_mask]), m


def forecast_target(X, y, params=None):
    """returns: forecast vector for all hours (OOF for 2021-24, true out-of-sample for 2025)"""
    fc = np.full(N, np.nan)
    years = [2022, 2023, 2024, 2025]
    for yr in years:                                    # leave-one-year-out inside the training period
        tr = np.isin(YEAR, [a for a in years if a != yr])
        pr = YEAR == yr
        fc[pr], _ = fit_predict(X, y, tr, pr, params)
    fc[TE], model = fit_predict(X, y, TR, TE, params)
    return fc, model


def trend_scale(y):
    """log-linear growth trend fitted on TRAINING days only, extrapolated to the test year; mean 1 over training"""
    dm = pd.Series(y).groupby(np.arange(N) // 24).mean().values
    tr_days = np.where(TR[::24][:len(dm)])[0]
    b = np.polyfit(tr_days, np.log(dm[tr_days]), 1)[0]
    sc = np.exp(b * np.arange(N) / 24.0)
    return sc / sc[TR].mean()


def forecast_price(y, nat):
    """IEX day-ahead price: model the ratio to the trailing 30-day level (known at bid time); yesterday's prices are known
    (results for D-1 are published before the 10:00 bid gate on D-1)."""
    lvl = pd.Series(y).shift(24).rolling(24 * 30, min_periods=24 * 7).mean().values
    yn = y / lvl
    s = pd.Series(yn)
    F = dict(hour=HOUR, dow=DOW, month=MONTH, doy=DOY, hol=dg.IS_HOL, dayoff=dayoff,
             lag24=s.shift(24).values, lag48=s.shift(48).values, lag168=s.shift(168).values,
             d24=s.shift(24).rolling(24).mean().values, wk_mean=np.mean([s.shift(24 * k).values for k in range(1, 8)], axis=0),
             loglvl=np.log(lvl), T_fc=nat['T'], cl_fc=nat['cl'], w_fc=nat['w'], heat_fc=np.maximum(nat['T'] - 31, 0))
    fc_n, model = forecast_target(pd.DataFrame(F), yn, PARAMS_PRICE)
    return fc_n * lvl, model


def forecast_detrended(df, y, kind, Xfun):
    sc = trend_scale(y)
    yn = y / sc
    fc, model = forecast_target(Xfun(yn), yn)
    return fc * sc, model


# ------------------------------------------------------------------------------------------------------------
def wape(a, f):
    return np.abs(a - f).sum() / np.abs(a).sum()


def main():
    t0 = time.time()
    out, tbl, eps_rows = {}, [], []
    nat = dict(T=np.mean([FR[s]['T_fc'].values for s in ['DEL', 'HYD', 'MUM']], axis=0),
               cl=np.mean([FR[s]['cloud_fc'].values for s in SHORT], axis=0),
               w=np.mean([FR[s]['wind_fc'].values for s in ['MUM', 'BLR', 'MAA']], axis=0))
    est, naive, actual = {}, {}, {}
    eps_hat = {}
    # 1) elasticities & de-priced demand (what a forecaster can observe)
    for s in SHORT:
        df = FR[s]
        e_dc = estimate_elasticity(df, 'dc_obs'); e_sw = estimate_elasticity(df, 'sw_obs')
        tr_dc = np.array([dg.EPS_DC[a:b].mean() for a, b in BLOCKS]); tr_sw = np.array([dg.EPS_SW[a:b].mean() for a, b in BLOCKS])
        for k in range(4):
            eps_rows.append((s, 'DC', f'{BLOCKS[k][0]:02d}-{BLOCKS[k][1]:02d}h', tr_dc[k], e_dc[k]))
            eps_rows.append((s, 'Swap', f'{BLOCKS[k][0]:02d}-{BLOCKS[k][1]:02d}h', tr_sw[k], e_sw[k]))
        bh = block_of(np.arange(24))
        eps_hat[s] = (e_dc[bh], e_sw[bh])
        est[(s, 'dc')] = df['dc_obs'].values * np.power(df['m_hist'].values, -eps_hat[s][0][HOUR])
        est[(s, 'sw')] = df['sw_obs'].values * np.power(df['m_hist'].values, -eps_hat[s][1][HOUR])
        actual[(s, 'dc')], actual[(s, 'sw')] = est[(s, 'dc')], est[(s, 'sw')]
    # 2) forecasts per target
    raw = {}
    for s in SHORT:
        df = FR[s]
        for kind in ('dc', 'sw'):
            y = est[(s, kind)]
            raw[(s, kind)], _ = forecast_detrended(df, y, kind, lambda yn, df=df, kind=kind: feats(df, yn, kind))
            naive[(s, kind)] = pd.Series(y).shift(168).values
        y = df['re'].values
        raw[(s, 're')], _ = forecast_target(feats(df, y, 're'), y)
        naive[(s, 're')] = pd.Series(y).shift(24).values
        actual[(s, 're')] = y
        print(s, 'done', round(time.time() - t0), 's', flush=True)
    yd = FR['DEL']['dam'].values
    raw[('NAT', 'dam')], _ = forecast_price(yd, nat)
    naive[('NAT', 'dam')] = pd.Series(yd).shift(24).values
    actual[('NAT', 'dam')] = yd
    # 3) hierarchical reconciliation (cluster -> network) for DC and swap demand
    rec = {}
    for kind in ('dc', 'sw'):
        tot_y = sum(est[(s, kind)] for s in SHORT)
        mkX = lambda yn: pd.DataFrame(dict(hour=HOUR, dow=DOW, month=MONTH, doy=DOY, hol=dg.IS_HOL, dayoff=dayoff, t=np.arange(N) / 8760.0,
                                           T_fc=nat['T'], heat_fc=np.maximum(nat['T'] - 30, 0), **lagfeat(yn)))
        raw[('NET', kind)], _ = forecast_detrended(None, tot_y, kind, mkX)
        actual[('NET', kind)] = tot_y
        naive[('NET', kind)] = pd.Series(tot_y).shift(168).values
        nodes = SHORT + ['NET']
        Y = np.column_stack([actual[(n, kind)] for n in nodes]); Fh = np.column_stack([raw[(n, kind)] for n in nodes])
        ok = TR & np.isfinite(Fh).all(axis=1)
        var = ((Y - Fh)[ok] ** 2).mean(axis=0)
        S = np.vstack([np.eye(5), np.ones((1, 5))])
        Wi = np.diag(1 / var)
        G = np.linalg.inv(S.T @ Wi @ S) @ S.T @ Wi
        Fr = (G @ np.nan_to_num(Fh).T).T                  # reconciled bottom-level forecasts
        for j, s in enumerate(SHORT):
            rec[(s, kind)] = np.where(np.isfinite(Fh[:, j]), Fr[:, j], np.nan)
        rec[('NET', kind)] = np.where(np.isfinite(Fh[:, 0]), Fr.sum(axis=1), np.nan)
    # 4) accuracy table (2025, out-of-sample)
    def row(name, node, kind, a, nv, f, fr_=None):
        m = TE & np.isfinite(nv)
        r = dict(series=name, node=node, kind=kind, naive_wape=wape(a[m], nv[m]), lgbm_wape=wape(a[m], f[m]))
        r['rec_wape'] = wape(a[m], fr_[m]) if fr_ is not None else np.nan
        r['bias_pct'] = (f[m] - a[m]).sum() / a[m].sum()
        return r
    for s in SHORT:
        for kind in ('dc', 'sw'):
            tbl.append(row(f'{NAME[s]} {kind.upper()} demand', s, kind, actual[(s, kind)], naive[(s, kind)], raw[(s, kind)], rec[(s, kind)]))
        tbl.append(row(f'{NAME[s]} RE output', s, 're', actual[(s, 're')], naive[(s, 're')], raw[(s, 're')]))
    for kind in ('dc', 'sw'):
        tbl.append(row(f'Network {kind.upper()} demand', 'NET', kind, actual[('NET', kind)], naive[('NET', kind)], raw[('NET', kind)], rec[('NET', kind)]))
    tbl.append(row('IEX DAM price', 'NAT', 'dam', actual[('NAT', 'dam')], naive[('NAT', 'dam')], raw[('NAT', 'dam')]))
    acc = pd.DataFrame(tbl)
    acc.to_csv('../out/tables/forecast_accuracy.csv', index=False)
    pd.DataFrame(eps_rows, columns=['cluster', 'product', 'block', 'true_eps', 'estimated_eps']).to_csv('../out/tables/elasticity_estimates.csv', index=False)
    # 5) final forecast arrays + error library (30-h windows of out-of-fold residuals, 2021-2024)
    fcs = {}
    day_starts = [i for i in range(0, N - 30, 24) if YEAR[i] <= 2025 and YEAR[i + 29] <= 2025 and i >= 24 * 14]
    for s in SHORT:
        f = {'dc': rec[(s, 'dc')], 'sw': rec[(s, 'sw')], 're': raw[(s, 're')], 'dam': raw[('NAT', 'dam')]}
        for k in f:
            f[k] = np.where(np.isfinite(f[k]), f[k], np.nan)
        f['dc'] = np.maximum(f['dc'], 0.05); f['sw'] = np.maximum(f['sw'], 0.02); f['re'] = np.maximum(f['re'], 0)
        f['dam'] = np.clip(f['dam'], 0.0, PRICE_CAP)
        lib = dict(month=[], wkend=[], dc=[], sw=[], re=[], dam=[])
        a = dict(dc=actual[(s, 'dc')], sw=actual[(s, 'sw')], re=actual[(s, 're')], dam=actual[('NAT', 'dam')])
        for i in day_starts:
            sl = slice(i, i + 30)
            if not all(np.isfinite(f[k][sl]).all() for k in f):
                continue
            lib['month'].append(MONTH[i]); lib['wkend'].append(int(DOW[i] >= 5))
            lib['dc'].append(np.log(np.maximum(a['dc'][sl], 1e-3) / f['dc'][sl]))
            lib['sw'].append(np.log(np.maximum(a['sw'][sl], 1e-3) / f['sw'][sl]))
            lib['re'].append(a['re'][sl] - f['re'][sl])
            lib['dam'].append(a['dam'][sl] - f['dam'][sl])
        fcs[s] = {k: np.nan_to_num(v, nan=0.0) for k, v in f.items()}
        fcs[s]['lib'] = {k: np.array(v) for k, v in lib.items()}
        fcs[s]['eps_hat'] = eps_hat[s]
    pickle.dump(fcs, open('../data/forecasts.pkl', 'wb'))
    # 6) interval calibration of the scenario library on 2025 (P10-P90 coverage of bootstrapped bands, DC demand)
    cov = []
    rng = np.random.default_rng(1)
    for s in SHORT:
        lib = fcs[s]['lib']['dc'][:, :24]
        lo, hi = np.percentile(lib, 10, axis=0), np.percentile(lib, 90, axis=0)
        i0 = np.where(TE)[0][0]
        a = actual[(s, 'dc')][i0:i0 + 8760].reshape(-1, 24); f = fcs[s]['dc'][i0:i0 + 8760].reshape(-1, 24)
        cov.append((s, ((np.log(a / f) >= lo) & (np.log(a / f) <= hi)).mean()))
    pd.DataFrame(cov, columns=['cluster', 'P10_P90_coverage']).to_csv('../out/tables/interval_coverage.csv', index=False)
    print(acc.round(3).to_string())
    print(pd.DataFrame(cov), time.time() - t0)


if __name__ == '__main__':
    main()
