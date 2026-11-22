"""5-year hourly dataset (23 Nov 2021 - 22 Nov 2026) for the 5 metro clusters. The IEX day-ahead price is REAL (iex_dam_5yr_hourly.csv);
weather, renewable output, demand and customer-price logs are synthetic.
Mimics the data the case says we can assume: weather, renewable generation, cluster demand, IEX price,
customer price logs and battery state-of-health records. Physically structured (not random noise)."""
import os
import numpy as np, pandas as pd
from params import *

MODE = os.environ.get('DATA_MODE', 'real')        # 'real' = IEX + ERA5 files;  'synthetic' = fully synthetic weather and price
START, END = '2021-11-23', '2026-11-22 23:00'
IDX = pd.date_range(START, END, freq='h')
N = len(IDX)
HOUR = IDX.hour.values
DOY = IDX.day_of_year.values
DOW = IDX.day_of_week.values
_blk = (len(IDX) - 1 - np.arange(len(IDX))) // 8760
YEAR = 2026 - _blk                      # 'operating year' label: years end 22 Nov; 2026 = TEST year (last 365 days)
TESTM = YEAR == 2026
TEST_Y = 2026
MONTH = IDX.month.values

HOLIDAYS = pd.to_datetime([
    '2021-01-26', '2021-03-29', '2021-08-15', '2021-10-02', '2021-11-04', '2021-12-25',
    '2022-01-26', '2022-03-18', '2022-08-15', '2022-10-02', '2022-10-24', '2022-12-25',
    '2023-01-26', '2023-03-08', '2023-08-15', '2023-10-02', '2023-11-12', '2023-12-25',
    '2024-01-26', '2024-03-25', '2024-08-15', '2024-10-02', '2024-11-01', '2024-12-25',
    '2025-01-26', '2025-03-14', '2025-08-15', '2025-10-02', '2025-10-20', '2025-12-25',
    '2026-01-26', '2026-03-04', '2026-08-15', '2026-10-02', '2026-11-08'])
IS_HOL = np.isin(IDX.normalize(), HOLIDAYS).astype(float)


def ar1(n, phi, sd, rng):
    """stationary AR(1) with marginal std sd"""
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n)
    x = np.zeros(n)
    x[0] = rng.normal(0, sd)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def bump(h, mu, sd):
    d = np.abs(h - mu)
    d = np.minimum(d, 24 - d)
    return np.exp(-0.5 * (d / sd) ** 2)


def _norm(v):
    return v / v.mean()


H24 = np.arange(24, dtype=float)
SHAPE_DC = {
    'retail': _norm(0.20 + 0.6 * bump(H24, 10, 1.8) + 1.0 * bump(H24, 19.5, 2.2) + 0.35 * bump(H24, 14, 2.0)),
    'ride': _norm(0.25 + 0.9 * bump(H24, 9, 1.8) + 1.0 * bump(H24, 18.5, 2.2) + 0.4 * bump(H24, 13, 2.0) + 0.2 * bump(H24, 23, 2.0)),
    'logi': _norm(0.30 + 0.9 * bump(H24, 23.5, 3.0) + 0.9 * bump(H24, 2.0, 2.5) + 0.5 * bump(H24, 13.5, 2.0)),
}
SHAPE_SW = {
    'retail': _norm(0.20 + 1.0 * bump(H24, 18.5, 2.0)),
    'ride': _norm(0.15 + 1.0 * bump(H24, 8, 1.8) + 0.8 * bump(H24, 14.5, 2.5) + 0.9 * bump(H24, 18, 2.0)),
    'logi': _norm(0.15 + 1.0 * bump(H24, 6.5, 1.5) + 0.9 * bump(H24, 13.5, 2.0) + 0.5 * bump(H24, 17.5, 2.0)),
}


def eps_by_hour(seg, shapes):
    """energy-weighted own-price elasticity for each hour-of-day"""
    num = sum(seg[k][0] * shapes[k] * seg[k][1] for k in seg)
    den = sum(seg[k][0] * shapes[k] for k in seg)
    return num / den


EPS_DC = eps_by_hour(SEG_DC, SHAPE_DC)     # 24-vector, true
EPS_SW = eps_by_hour(SEG_SW, SHAPE_SW)


def demand_response(ref, m, eps_h, hour, rho=RHO_TRUE):
    """True customer response to hourly price multiplier m (vs reference tariff): own-price elasticity +
    partial recapture (rho) of suppressed volume in the next 1-4 hours."""
    eps = eps_h[hour]
    d0 = ref * np.power(m, eps)
    delta = ref - d0
    rec = np.zeros_like(delta)
    for j, w in enumerate(RECAP_W, start=1):
        rec[j:] += rho * w * delta[:-j]
    return np.maximum(d0 + rec, 0.0)


def tou_multiplier(hour):
    """baseline static ToU price multiplier: +10 % evening peak, -10 % night, else 1"""
    m = np.ones(len(hour))
    m[(hour >= 18) & (hour < 22)] = 1.10
    m[(hour >= 0) & (hour < 6)] = 0.90
    return m


# Standard 3 MW onshore turbine curve at 100 m hub height: wind speed (m/s) -> capacity factor (cut-in 3, rated 12, cut-out 25 m/s)
_WS_PTS = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 25, 25.1, 50]
_CF_PTS = [0, 0, 0.04, 0.11, 0.21, 0.35, 0.52, 0.70, 0.85, 0.95, 1.0, 1.0, 0.0, 0.0]


def load_era5(code):
    """real ERA5 hourly weather (Open-Meteo archive, local time): temperature, cloud, 100 m wind (km/h -> m/s), radiation.
    Capacity factors follow the supplied method: solar_cf = ghi/1000 * (0.946 - 0.00344*T - 0.0001*ghi);
    wind_cf = 3 MW onshore turbine curve of the 100 m wind speed.
    ERA5 ends 20 Nov 2026; the last 48 h of the index repeat the final available day."""
    x = pd.read_csv(f'../data/era5_{code}.csv', parse_dates=['time']).set_index('time').sort_index()
    x = x[~x.index.duplicated()]
    if 'solar_cf' not in x.columns:
        x['solar_cf'] = np.clip(x['shortwave_radiation'] / 1000.0 * (0.946 - 0.00344 * x['temperature_2m'] - 0.0001 * x['shortwave_radiation']), 0, 1)
        x['wind_cf'] = np.interp(x['wind_speed_100m'].values / 3.6, _WS_PTS, _CF_PTS)      # Open-Meteo wind is km/h
    out = pd.DataFrame({'T': x['temperature_2m'], 'cloud': x['cloud_cover'] / 100.0, 'wind': x['wind_speed_100m'] / 3.6,
                        'ghi': x['shortwave_radiation'], 'sol': x['solar_cf'], 'wcf': x['wind_cf']}).reindex(IDX)
    miss = out.index[out['T'].isna()]
    if len(miss):
        last = out.dropna().index[-24:]
        for t in miss:
            out.loc[t] = out.loc[last[t.hour == last.hour]].iloc[0].values
    assert not out.isna().any().any()
    return out


def make_base(seed=2025):
    rng = np.random.default_rng(seed)
    out = {}
    # ---------- national weather proxy (for IEX price) ----------
    nat_T = None
    clusters = {}
    for s in SHORT:
        c = CFG[s]
        # REAL ERA5 reanalysis weather (Open-Meteo archive, local time): temperature, cloud, 100 m wind, shortwave radiation
        if MODE == 'real':
            W = load_era5(s)
            T, cloud, v = W['T'].values, W['cloud'].values, W['wind'].values
            sol, wcf = W['sol'].values, W['wcf'].values
        else:
            T, cloud, v, sol, wcf = synth_weather(c, rng)
        # day-ahead weather forecasts (NWP-like errors on top of the reanalysis 'truth')
        T_fc = T + ar1(N, 0.9, 1.4, rng)
        cl_fc = np.clip(cloud + ar1(N, 0.9, 0.15, rng), 0, 1)
        v_fc = np.clip(v + ar1(N, 0.9, 1.5, rng), 0, 40)
        # ---------- demand shapes ----------
        wk = (DOW >= 5).astype(float)
        hol = IS_HOL
        dayoff = np.clip(wk + hol, 0, 1)
        dc = np.zeros(N)
        for k, (share, _) in SEG_DC.items():
            f = {'retail': 1 + 0.08 * dayoff, 'ride': 1 - 0.08 * dayoff, 'logi': 1 - 0.28 * dayoff}[k]
            dc += share * SHAPE_DC[k][HOUR] * f
        sw = np.zeros(N)
        for k, (share, _) in SEG_SW.items():
            f = {'retail': 1 + 0.05 * dayoff, 'ride': 1 - 0.05 * dayoff, 'logi': 1 - 0.25 * dayoff}[k]
            sw += share * SHAPE_SW[k][HOUR] * f
        yr = (np.arange(N) - (N - 4380)) / 8760.0
        growth = (1 + GROWTH) ** yr
        heat_dc = 1 + 0.016 * np.maximum(T - 30, 0)
        heat_sw = 1 + 0.010 * np.maximum(T - 30, 0)
        rain = np.where(cloud > 0.85, 0.93, 1.0)
        seas_d = 1 + 0.04 * np.cos(2 * np.pi * (DOY - 330) / 365)
        nz = np.exp(ar1(N, 0.75, 0.09, rng) + np.repeat(ar1(N // 24 + 1, 0.5, 0.06, rng), 24)[:N])
        dc_unit = dc * growth * heat_dc * rain * seas_d * nz
        nz2 = np.exp(ar1(N, 0.75, 0.10, rng) + np.repeat(ar1(N // 24 + 1, 0.5, 0.06, rng), 24)[:N])
        sw_unit = sw * growth * heat_sw * rain * seas_d * nz2
        m25 = TESTM
        clusters[s] = pd.DataFrame(dict(
            T=T, cloud=cloud, wind=v, T_fc=T_fc, cloud_fc=cl_fc, wind_fc=v_fc,
            solar_cf=np.clip(sol, 0, 1), wind_cf=wcf,
            dc_unit=dc_unit / dc_unit[m25].mean(), sw_unit=sw_unit / sw_unit[m25].mean()), index=IDX)
    # ---------- IEX day-ahead price: REAL data ----------
    price = load_iex() if MODE == 'real' else synth_price(clusters, rng)
    for s in SHORT:
        clusters[s]['dam'] = price
    return clusters, price


def synth_weather(c, rng):
    """fully synthetic weather + renewable capacity factors for a city (used when DATA_MODE='synthetic')"""
    seas = c['Tm'] + c['Ta'] * np.sin(2 * np.pi * (DOY - 105) / 365.25)
    diur = (c['Td'] / 2) * np.sin(2 * np.pi * (HOUR - 9) / 24)
    syn = ar1(N, 0.985, 1.6, rng)
    hw = np.zeros(N)
    for y in range(2022, 2027):                     # 2-4 heatwave episodes per year in Mar-Jun
        for _ in range(rng.integers(2, 5)):
            st = pd.Timestamp(f'{y}-03-20') + pd.Timedelta(days=int(rng.integers(0, 90)))
            ln = int(rng.integers(3, 9))
            a = np.searchsorted(IDX, st)
            hw[a:a + ln * 24] += rng.uniform(3.0, 6.0) * (0.6 + 0.4 * np.sin(np.pi * np.arange(min(ln * 24, max(N - a, 1))) / (ln * 24)))[:max(min(ln * 24, N - a), 0)]
    T = seas + diur + syn + hw
    m0, m1 = c['monsoon']
    mon = np.exp(-0.5 * ((((DOY / 30.4) - (m0 + m1) / 2 + 6) % 12 - 6) / ((m1 - m0) / 2.2)) ** 2)
    cm = np.clip(c['cloud_base'] + c['cloud_amp'] * mon, 0.05, 0.92)
    cloud = 1 / (1 + np.exp(-(np.log(cm / (1 - cm)) + ar1(N, 0.96, 0.9, rng) + ar1(N, 0.7, 0.35, rng))))
    w0, w1 = c['windy']
    wm = np.exp(-0.5 * ((((DOY / 30.4) - (w0 + w1) / 2 + 6) % 12 - 6) / ((w1 - w0) / 2.0)) ** 2)
    g = ar1(N, 0.975, 1.0, rng)
    lull = np.zeros(N)
    for y in range(2022, 2027):
        for _ in range(rng.integers(3, 7)):
            a = np.searchsorted(IDX, pd.Timestamp(f'{y}-01-01') + pd.Timedelta(days=int(rng.integers(0, 360))))
            lull[a:a + int(rng.integers(2, 5)) * 24] -= rng.uniform(1.0, 1.8)
    v = np.clip(np.exp(np.log(6.8) + 0.30 * wm + 0.20 * np.cos(2 * np.pi * (HOUR - 20) / 24) + 0.30 * (g + lull)), 0, 28)
    lat = np.deg2rad(c['lat'])
    dec = np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + DOY) / 365)
    ha = np.deg2rad(15 * (HOUR + 0.5 - 12))
    cs = np.maximum(np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha), 0)
    tcell = T + 20 * cs
    sol = np.clip(0.86 * cs ** 0.85 * (1 - 0.78 * cloud ** 1.4) * (1 - 0.004 * np.maximum(tcell - 25, 0)), 0, 1)
    wcf = np.where(v < 3, 0, np.where(v < 12, ((v - 3) / 9) ** 3, np.where(v < 25, 1.0, 0.0)))
    return T, cloud, v * 3.6 / 3.6, sol, wcf


def synth_price(clusters, rng):
    """fully synthetic IEX-like day-ahead price (Rs/kWh), correlated with the synthetic weather"""
    Tn = np.mean([clusters[s]['T'].values for s in ['DEL', 'HYD', 'MUM']], axis=0)
    wn = np.mean([clusters[s]['wind_cf'].values for s in ['MUM', 'BLR', 'MAA']], axis=0)
    sn = np.mean([clusters[s]['solar_cf'].values for s in ['DEL', 'HYD', 'BLR']], axis=0)
    base = {2021: 4.9, 2022: 7.4, 2023: 6.1, 2024: 5.6, 2025: 5.4, 2026: 5.6}
    lvl = np.array([base[y] for y in YEAR])
    fuel = np.exp(np.repeat(ar1(N // 24 + 1, 0.995, 0.10, rng), 24)[:N])
    shape = (1.0 + 0.40 * bump(HOUR, 20.5, 1.8) + 0.12 * bump(HOUR, 8.5, 1.5)
             - 0.30 * bump(HOUR, 12.5, 2.6) - 0.10 * bump(HOUR, 3.5, 2.5))
    seasp = 1 + 0.12 * np.exp(-0.5 * ((DOY - 150) / 40) ** 2) - 0.10 * np.exp(-0.5 * ((DOY - 215) / 35) ** 2)
    heat = 0.20 * np.maximum(Tn - 31, 0) / 4
    solar_dip = 0.5 * sn * (YEAR - 2020) / 6
    nzp = ar1(N, 0.9, 0.12, rng) + np.repeat(ar1(N // 24 + 1, 0.6, 0.12, rng), 24)[:N]
    price = lvl * fuel * shape * seasp * (1 + heat) * np.exp(nzp) * (1 - 0.30 * wn * 0.4) - solar_dip
    spikes = np.zeros(N)
    for _ in range(110):
        a = int(rng.integers(0, N - 6))
        spikes[a:a + int(rng.integers(2, 7))] += rng.uniform(1.5, 4.0)
    return np.clip(price + spikes * (0.5 + bump(HOUR, 20, 4)), 0.3, 10.0)


def load_iex(path='../data/iex_dam_5yr_hourly.csv'):
    """real IEX DAM market clearing price (Rs/MWh in the file) -> Rs/kWh on the model's hourly index.
    Hour 1 of the file = 00:00-01:00.  A 30-day gap (6 Apr - 5 May 2025, inside training) is filled with the
    same-hour-of-day mean of the 14 days either side."""
    x = pd.read_csv(path)
    x['dt'] = pd.to_datetime(x['Date'], format='%d-%m-%Y') + pd.to_timedelta(x['Hour'] - 1, unit='h')
    s = x.set_index('dt')['MCP (Rs/MWh) *'].sort_index().astype(float)
    s = s[~s.index.duplicated()].reindex(IDX)
    miss = s.index[s.isna()]
    if len(miss):
        a, b = miss.min(), miss.max()
        ctx = pd.concat([s[a - pd.Timedelta(days=14):a - pd.Timedelta(hours=1)], s[b + pd.Timedelta(hours=1):b + pd.Timedelta(days=14)]])
        prof = ctx.groupby(ctx.index.hour).mean()
        s[miss] = prof.reindex(miss.hour).values
    assert not s.isna().any()
    return s.values / 1000.0


def apply_calib(clusters, calib, fixed_re_mw=None):
    """scale to MW, size the RE contracts to 35 % of energy and create customer-price history"""
    frames = {}
    rng = np.random.default_rng(7)
    for s in SHORT:
        c, k = CFG[s], calib[s]
        df = clusters[s].copy()
        df['dc_ref'] = k['dc_mw'] * df['dc_unit']          # MWh requested at reference price
        df['sw_ref'] = k['sw_mw'] * df['sw_unit']
        m25 = TESTM
        bus_mw = (df.loc[m25, 'dc_ref'].mean() / ETA_DC + df.loc[m25, 'sw_ref'].mean() / ETA_SW) / (1 - DIST_LOSS)
        re_e = RE_SHARE * bus_mw * 8760 / 8760                       # target mean MW of RE energy
        sol_cf, win_cf = df.loc[m25, 'solar_cf'].mean(), df.loc[m25, 'wind_cf'].mean()
        sol_mw = c['solar'] * re_e / sol_cf
        win_mw = (1 - c['solar']) * re_e / win_cf if c['solar'] < 1 else 0.0
        df['re'] = sol_mw * df['solar_cf'] + win_mw * df['wind_cf']
        df['re_mw_sol'], df['re_mw_win'] = sol_mw, win_mw
        # reference tariff: market-determined, daily random walk linked to IEX average
        day = df.index.normalize()
        dmean = df.groupby(day)['dam'].mean()
        prev = dmean.shift(1).fillna(dmean.iloc[0])                  # yesterday's average IEX price
        dayavg = day.map(prev).values.astype(float)
        n_days = day.nunique()
        wk = np.cumsum(rng.normal(0, 0.006, n_days))
        pr = np.exp(np.repeat(wk - wk.mean(), 24)[:N] * 0.6)
        link = 1 + 0.10 * (dayavg - 4.9) / 4.9
        df['p_ref_dc'] = P_DC * c['price_f'] * pr * link
        df['p_ref_sw'] = P_SW * c['price_f'] * pr * link
        # historical customer price multiplier = static ToU x experimentation noise (A/B)
        mt = tou_multiplier(HOUR) * np.exp(rng.normal(0, 0.06, N))
        df['m_hist'] = np.clip(mt, 0.8, 1.2)
        df['dc_obs'] = demand_response(df['dc_ref'].values, df['m_hist'].values, EPS_DC, HOUR) * np.exp(rng.normal(0, 0.03, N))
        df['sw_obs'] = demand_response(df['sw_ref'].values, df['m_hist'].values, EPS_SW, HOUR) * np.exp(rng.normal(0, 0.03, N))
        df['hol'] = IS_HOL
        frames[s] = df
    return frames


def soh_records(seed=11):
    """monthly battery state-of-health audit records (stationary BESS & swap packs) per cluster from a hidden
    ground-truth ageing law (calendar sqrt-t x Arrhenius + cycle term) + measurement noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in SHORT:
        Tm = CFG[s]['Tm']
        for kind, cyc_per_day, fade_cal, fade_cyc in [('bess', 0.9, 0.040, 0.2 / 6000 * 100), ('pack', 1.8, 0.050, 0.2 / 2500 * 100)]:
            for u in range(6):                                   # six audited units per cluster per kind
                age0 = rng.uniform(0, 1.5)
                for mth in range(0, 49, 1):
                    t = age0 + mth / 12
                    efc = cyc_per_day * 365 * t * rng.uniform(0.9, 1.1)
                    Tu = Tm + rng.normal(2.0, 0.5)
                    arr = 2 ** ((Tu - 25) / 10)
                    fade = fade_cal * arr * np.sqrt(t) + fade_cyc * efc / 100 * (1 + 0.03 * max(Tu - 25, 0))
                    rows.append((s, kind, u, mth, t, efc, Tu, 1 - fade + rng.normal(0, 0.004)))
    return pd.DataFrame(rows, columns=['cluster', 'kind', 'unit', 'month', 'age_yr', 'efc', 'temp', 'soh'])


if __name__ == '__main__':
    cl, price = make_base()
    fr = apply_calib(cl, DEFAULT_CALIB)
    for s in SHORT:
        x = fr[s]; m = TESTM
        print(s, 'dc', x.loc[m, 'dc_ref'].mean().round(1), 'sw', x.loc[m, 'sw_ref'].mean().round(1), 'RE', x.loc[m, 're'].mean().round(1),
              'solCF', x.loc[m, 'solar_cf'].mean().round(3), 'windCF', x.loc[m, 'wind_cf'].mean().round(3), 'Tmax', x.loc[m, 'T'].max().round(1))
    for y in sorted(set(YEAR)):
        print(y, 'n', (YEAR == y).sum(), 'DAM avg', price[YEAR == y].mean().round(2), 'max', price[YEAR == y].max().round(2))
