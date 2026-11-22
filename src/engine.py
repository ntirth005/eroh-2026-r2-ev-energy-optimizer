"""Operating simulator: hourly physical + financial model of one metro cluster, with pluggable policies.
policy keys:
  fc      : 'naive' | 'ml'            demand/RE/price forecasts used for planning
  da      : 'rule' | 'det' | 'stoch'  day-ahead bid (q, z) logic
  price   : 'tou' | 'opt'             customer pricing (static ToU vs elasticity-aware optimiser)
  rt      : 'rule' | 'mpc'            intra-day storage / pack dispatch
  sell_re : bool                      monetise surplus RE on the exchange
  steer   : bool                      hub steering: optimiser pays a loyalty credit to move overflow to idle hubs
  rebal   : bool                      forecast-driven rebalancing of rack inventory between stations
  pi      : float                     service-level premium Rs/MWh
"""
import numpy as np, pandas as pd, time
from scipy.stats import norm
from params import *
import datagen as dg
import lp as L

FULL = L.FULL
N_HUB = 20


def tariff(cl):
    return CFG[cl]['discom'] * 1000.0 * TOD            # Rs/MWh by hour-of-day


def resp(ref, m, hours, eps_h, rho, prev_delta=None):
    """price response with partial recapture of suppressed volume in the next 1-4 h.
    prev_delta: suppressed volume of the 4 hours preceding the window (carry-in). returns (demand, delta)"""
    eps = eps_h[hours]
    d0 = ref * np.power(m, eps)
    delta = ref - d0
    n = len(delta)
    full = np.concatenate([prev_delta if prev_delta is not None else np.zeros(4), delta])
    rec = np.zeros(n)
    for j, w in enumerate(RECAP_W, start=1):
        rec += rho * w * full[4 - j:4 - j + n]
    return np.maximum(d0 + rec, 0.0), delta


def hub_weights(sd):
    """demand share of each of 20 equal-capacity hubs: lognormal dispersion sd (quantile grid, sums to 1)"""
    z = norm.ppf((np.arange(N_HUB) + 0.5) / N_HUB)
    w = np.exp(sd * z)
    return w / w.sum()


def hub_served(D, Ctot, w, sigma, ret_red=False):
    """DC energy servable when cluster demand D is spread over hubs with weights w (equal chargers, total Ctot).
    Hot hubs overflow while others idle; steering redirects a share sigma of the overflow to hubs with slack.
    All of D, Ctot, sigma broadcast against each other."""
    D = np.asarray(D, dtype=float)[..., None]
    c = np.asarray(Ctot, dtype=float)[..., None] / N_HUB
    sg = np.asarray(sigma, dtype=float)
    dj = D * w
    s = np.minimum(dj, c).sum(-1)
    over = np.maximum(dj - c, 0).sum(-1)
    slack = np.maximum(c - dj, 0).sum(-1)
    red = sg * np.minimum(over, slack)
    return (s + red, red) if ret_red else s + red


def swap_served(D, Csw, w, beta):
    """swap energy servable per hour: demand spread over stations with weights w; each station can hand out a share of
    the total hourly swap capability Csw in proportion to its rack inventory.  Rebalancing (beta) moves inventory
    toward stations in proportion to their demand weight."""
    D = np.asarray(D, dtype=float)[..., None]
    sj = (1 - beta) / N_HUB + beta * w
    return np.minimum(D * w, Csw * sj).sum(-1)


def sig_of(delta):
    """share of overflow redirected for a loyalty credit of `delta` (fraction of tariff, paid on redirected energy)"""
    return SIG_MAX * (1.0 - np.exp(-np.asarray(delta, dtype=float) / DELTA0))


class Cluster:
    def __init__(self, cl, df, calib, hw=None, N=None, grid_scale=1.0):
        self.cl, self.c = cl, CFG[cl]
        self.df = df
        self.hw = hw or L.HW()
        cap = self.c['cap']
        self.Pdc, self.Psw = DC_SHARE * cap, SW_SHARE * cap
        self.E = float(self.c['bess']); self.Pb = BESS_C * self.E
        self.N = float(N if N is not None else self.c['pack'])
        self.G = calib[cl]['grid'] * grid_scale
        self.hubw = hub_weights(calib[cl].get('hub_sd', 0.6))
        self.wsw = hub_weights(calib[cl].get('sw_sd', 0.6))
        self.Csw = TURN_FRAC * FULL * self.N
        self.tar = tariff(cl)
        a = lambda k: df[k].values.astype(float)
        self.T, self.T_fc = a('T'), a('T_fc')
        self.dc_ref, self.sw_ref, self.re = a('dc_ref'), a('sw_ref'), a('re')
        self.dam = a('dam') * 1000.0                     # Rs/MWh
        self.pdc_ref, self.psw_ref = a('p_ref_dc') * 1000.0, a('p_ref_sw') * 1000.0   # Rs/MWh
        self.hour = df.index.hour.values
        self.dc_obs, self.sw_obs = a('dc_obs'), a('sw_obs')
        self.Gh = a('G') if 'G' in df.columns else None
        self.psw_cap = CHARGER_AVAIL * self.Psw
        self.wear_t = bess_wear_rs_per_mwh(self.T)
        self.wear_fc = bess_wear_rs_per_mwh(self.T_fc)
        self.sw_var = SW_LOGISTICS * 1000.0 + pack_cycle_rs_per_mwh()
        self.beta = 0.0
        self.sigmax = SIG_MAX


# ------------------------------------------------------------------------------------------------------------
def naive_fc(C, i0, n):
    """seasonal-naive forecasts (what the incumbent process does): same hour last week for demand, yesterday for RE/price."""
    idx = np.minimum(np.arange(i0, i0 + n), len(C.T) - 1)
    return dict(dc=C.dc_obs[idx - 168], sw=C.sw_obs[idx - 168], re=C.re[idx - 24], dam=C.dam[idx - 24])


def ml_fc(C, fc, i0, n):
    idx = np.minimum(np.arange(i0, i0 + n), len(C.T) - 1)
    return dict(dc=fc['dc'][idx], sw=fc['sw'][idx], re=fc['re'][idx], dam=fc['dam'][idx] * 1000.0)   # Rs/kWh -> Rs/MWh


def scenarios(fc, f, i0, n, S, rng, month, weekend):
    """residual-day bootstrap: S historical forecast-error windows (same season / day-type) applied to the forecast"""
    lib = fc['lib']
    cand = np.where((np.abs(lib['month'] - month) <= 1) | (np.abs(lib['month'] - month) >= 11))[0]
    cand2 = cand[lib['wkend'][cand] == weekend]
    if len(cand2) >= S:
        cand = cand2
    pick = rng.choice(cand, size=S, replace=len(cand) < S)
    out = dict(dc=[], sw=[], re=[], dam=[])
    for k in pick:
        out['dc'].append(f['dc'] * np.exp(lib['dc'][k, :n]))
        out['sw'].append(f['sw'] * np.exp(lib['sw'][k, :n]))
        out['re'].append(np.maximum(f['re'] + lib['re'][k, :n], 0))
        out['dam'].append(np.clip(f['dam'] + lib['dam'][k, :n] * 1000.0, 0.0, PRICE_CAP * 1000.0))
    return {k: np.array(v) for k, v in out.items()}


# ------------------------------------------------------------------------------------------------------------
def choose_prices(C, Dhat, cap_dc, sup_dc, cap_sw, sup_sw, eps_hat, pol, mc_dc, mc_sw, pref_dc, pref_sw):
    """Elasticity-aware hourly price multiplier inside the +-20 % band, JOINTLY with the hub-steering credit.
    Day-average guard: sum_h (m_h - 1.04) d_h <= 0  (volume-weighted average <= +4 % vs market tariff), enforced with a
    Lagrange multiplier found by bisection:  maximise  profit_h(m, credit) - mu (m - 1.04) d_h(m)."""
    grid = np.linspace(1 - BAND, 1 + BAND, 17)
    cred = STEER_GRID if pol.get('steer') else np.array([0.0])
    sg = C.sigmax * (1.0 - np.exp(-cred / DELTA0)) if pol.get('steer') else np.zeros(1)
    res = []
    for kind in ('dc', 'sw'):
        D, eps = (Dhat[0], eps_hat[0]) if kind == 'dc' else (Dhat[1], eps_hat[1])
        D = np.maximum(D, 1e-6)
        dem = D[:, None] * np.power(grid[None, :], eps[:, None])                 # (24, K)
        if kind == 'dc':
            dK = np.broadcast_to(dem[:, :, None], (24, len(grid), len(cred)))
            served, red = hub_served(dK, cap_dc[:, None, None], C.hubw, sg[None, None, :], ret_red=True)
            qty = np.minimum(served, sup_dc[:, None, None])
            red = red * (qty / np.maximum(served, 1e-9))
            margin = pref_dc * grid[None, :, None] * (1 - VAR_OPEX) - mc_dc[:, None, None] + pol['pi']
            profit = margin * qty - cred[None, None, :] * pref_dc * red             # loyalty credit paid on redirected energy
            dd = dem[:, :, None] * np.ones((1, 1, len(cred)))
        else:
            qty0 = np.minimum(swap_served(dem, C.Csw, C.wsw, C.beta), sup_sw[:, None])
            profit = ((pref_sw * grid[None, :] * (1 - VAR_OPEX) - mc_sw[:, None] - C.sw_var + pol['pi']) * qty0)[:, :, None]
            dd = dem[:, :, None]
        gm = grid[None, :, None]
        lo, hi = 0.0, 4e5
        for _ in range(28):
            mu = 0.5 * (lo + hi)
            obj = profit - mu * (gm - 1.04) * dd
            flat = obj.reshape(24, -1).argmax(axis=1)
            k_m, k_c = np.unravel_index(flat, (len(grid), profit.shape[2]))
            m = grid[k_m]
            d_ = dem[np.arange(24), k_m]
            if (m * d_).sum() / d_.sum() > 1.04:
                lo = mu
            else:
                hi = mu
        obj = profit - hi * (gm - 1.04) * dd
        flat = obj.reshape(24, -1).argmax(axis=1)
        k_m, k_c = np.unravel_index(flat, (len(grid), profit.shape[2]))
        res.append((grid[k_m], cred[k_c] if kind == 'dc' else np.zeros(24)))
    return res[0][0], res[0][1], res[1][0]


def da_plan(C, st, i0, pol, rng, hw, ctx):
    """Day-ahead decisions at 10:00 of D-1 for the day starting at index i0: price multipliers, steering credit,
    DAM bids q, DISCOM schedule z.  Uses only information available then (forecast weather, yesterday's tariff)."""
    H = 30
    hrs = np.arange(H) % 24
    n = H
    f = ml_fc(C, ctx['fc'], i0, n) if pol['fc'] == 'ml' else naive_fc(C, i0, n)
    idx = np.minimum(np.arange(i0, i0 + n), len(C.T) - 1)
    pref_dc, pref_sw = C.pdc_ref[idx[0]], C.psw_ref[idx[0]]
    Tf = C.T_fc[idx]                                          # forecast temperature (NWP), not the realised one
    cap_dc = PEAK_FACTOR * CHARGER_AVAIL * C.Pdc * derate(Tf)
    tar = C.tar[hrs]
    eps_dc_h, eps_sw_h = ctx['eps_hat']
    prev = st['prev_delta']
    m_dc = np.ones(n); m_dc[(hrs >= 18) & (hrs < 22)] = 1.10; m_dc[hrs < 6] = 0.90          # static ToU
    m_sw = m_dc.copy()
    cred = np.zeros(n)
    plan = None
    n_iter = 2 if pol['price'] == 'opt' else 1
    S = ctx.get('S', 10) if pol['da'] == 'stoch' else 1
    if pol['da'] == 'stoch':
        sc = scenarios(ctx['fc'], f, i0, n, S, rng, C.df.index[i0].month, int(C.df.index[i0].dayofweek >= 5))
        sc['dc'][0], sc['sw'][0], sc['re'][0], sc['dam'][0] = f['dc'], f['sw'], f['re'], f['dam']
    else:
        sc = dict(dc=f['dc'][None], sw=f['sw'][None], re=f['re'][None], dam=f['dam'][None])
    for it in range(n_iter):
        sig = C.sigmax * (1 - np.exp(-cred / DELTA0)) if pol.get('steer') else np.zeros(n)
        dem_dc = np.array([resp(sc['dc'][w], m_dc, hrs, eps_dc_h, ctx['rho_hat'], prev['dc'])[0] for w in range(S)])
        dem_sw = np.array([resp(sc['sw'][w], m_sw, hrs, eps_sw_h, ctx['rho_hat'], prev['sw'])[0] for w in range(S)])
        if pol['da'] == 'rule' and pol['rt'] == 'rule':
            break
        sv = [hub_served(dem_dc[w], cap_dc, C.hubw, sig, ret_red=True) for w in range(S)]
        ub_dc = np.array([a_ for a_, _ in sv])
        rr = np.mean([b_ for _, b_ in sv], axis=0) / np.maximum(dem_dc.mean(axis=0), 1e-9)
        ub_sw = np.array([swap_served(dem_sw[w], C.Csw, C.wsw, C.beta) for w in range(S)])
        d = dict(E=C.E, Pb=C.Pb, N=C.N, G=C.G, Gh=None, dem_dc=dem_dc, dem_sw=dem_sw, ub_dc=ub_dc, ub_sw=ub_sw, psw_cap=C.psw_cap,
                 re=sc['re'], P=sc['dam'], Tar=tar, wear=C.wear_fc[idx],
                 pn_dc=pref_dc * m_dc * (1 - VAR_OPEX) - cred * pref_dc * rr,            # steering credit is a cost in the plan
                 pn_sw=pref_sw * m_sw * (1 - VAR_OPEX), pi=pol['pi'], sw_var=C.sw_var, cal=PACK_CAL_RS_MWH_H,
                 sell=1.0 if pol['sell_re'] else 0.0, init=(st['soc'], st['F']), kappa=pol.get('kappa', 0.3), alpha=0.8)
        Lbar = (sc['dam'].mean() + DAM_ADDER * 1000)
        d['nu_soc'] = 0.80 * Lbar
        d['nu_F'] = 0.80 * Lbar / (hw.eta_sw * (1 - DIST_LOSS))
        plan = L.solve(L.struct(H, S, hw, cvar=(pol['da'] == 'stoch')), d, hw)
        if plan is None:
            break
        if pol['price'] == 'opt' and it < n_iter - 1:
            Pm = sc['dam'].mean(axis=0)                                  # Rs/MWh
            cheap = np.minimum(tar, Pm + DAM_ADDER * 1000)
            re_m = sc['re'].mean(axis=0)
            load_est = (dem_dc.mean(axis=0) / hw.eta_dc + dem_sw.mean(axis=0) / hw.eta_sw) / (1 - DIST_LOSS)
            mc_bus = np.where(re_m > load_est, 0.0, cheap)
            mc_dc = mc_bus / (hw.eta_dc * (1 - DIST_LOSS))
            mc_sw = mc_bus / (hw.eta_sw * (1 - DIST_LOSS))
            sdc_s = np.mean([p['sdc'] for p in plan['sc']], axis=0)
            ssw_s = np.mean([p['ssw'] for p in plan['sc']], axis=0)
            sup_dc = np.where(sdc_s < 0.995 * ub_dc.mean(axis=0), sdc_s, 1e9)
            sup_sw = np.where(ssw_s < 0.995 * np.minimum(dem_sw.mean(axis=0), ub_sw.mean(axis=0)), ssw_s, 1e9)
            k24 = slice(0, 24)
            m1, c1, m2 = choose_prices(C, (f['dc'][k24], f['sw'][k24]), cap_dc[k24], sup_dc[k24], None, sup_sw[k24],
                                       (eps_dc_h[hrs[k24]], eps_sw_h[hrs[k24]]), pol, mc_dc[k24], mc_sw[k24], pref_dc, pref_sw)
            m_dc = np.concatenate([m1, np.ones(6)]); m_sw = np.concatenate([m2, np.ones(6)])
            cred = np.concatenate([c1, np.zeros(6)])
    sig = C.sigmax * (1 - np.exp(-cred / DELTA0)) if pol.get('steer') else np.zeros(n)
    out = dict(m_dc=m_dc, m_sw=m_sw, cred=cred, sig=sig, f=f, plan=plan, sc=sc)
    out['dd'] = resp(f['dc'], m_dc, hrs, eps_dc_h, ctx['rho_hat'], prev['dc'])[0]
    out['ds'] = resp(f['sw'], m_sw, hrs, eps_sw_h, ctx['rho_hat'], prev['sw'])[0]
    if plan is not None and pol['da'] in ('det', 'stoch'):
        out['q'], out['z'] = plan['q'], plan['z']
    else:        # incumbent rule: cheaper of DAM / DISCOM for the forecast net load that its network can physically DELIVER
        dd_del = hub_served(out['dd'], cap_dc, C.hubw, sig)
        ds_del = swap_served(out['ds'], C.Csw, C.wsw, C.beta)
        nl = np.maximum((dd_del / hw.eta_dc + ds_del / hw.eta_sw) / (1 - DIST_LOSS) - f['re'], 0)
        allow = np.maximum(C.G - f['re'], 0.0)
        nl = np.minimum(nl * pol.get('cov', 1.0), allow)
        # price-aware rule: each hour buy from the cheaper source judged on the price forecast it has
        # (naive = yesterday's same-hour IEX price; ML policies = model forecast) - NOT a fixed 50/50 split
        use_dam = (f['dam'] + DAM_ADDER * 1000) / (1 - TRANS_LOSS) < tar
        out['q'] = np.where(use_dam, nl / (1 - TRANS_LOSS), 0.0)
        out['z'] = np.where(use_dam, 0.0, nl)
    return out


def rt_mpc(C, st, h, day, i, pol, hw, ctx, rt_state):
    """receding-horizon dispatch for hour h of the day; returns planned (bc, bd, psw)."""
    H = int(min(6, 30 - h))
    sl = slice(h, h + H)
    q, z = day['q'][sl], day['z'][sl]
    e_dc, e_sw, d_re = rt_state['e_dc'], rt_state['e_sw'], rt_state['d_re']
    decay = 0.6 ** (np.arange(H) + 1)
    dem_dc = day['dd'][sl] * (1 + (e_dc - 1) * decay)
    dem_sw = day['ds'][sl] * (1 + (e_sw - 1) * decay)
    re = np.maximum(day['f']['re'][sl] + d_re * decay, 0)
    P = day['f']['dam'][sl]
    hrs = (h + np.arange(H)) % 24
    idx = np.minimum(i + np.arange(H), len(C.T) - 1)
    capT = PEAK_FACTOR * CHARGER_AVAIL * C.Pdc * derate(C.T_fc[idx])
    sg = day['sig'][sl]
    sv, rd = hub_served(dem_dc, capT, C.hubw, sg, ret_red=True)
    rr = rd / np.maximum(dem_dc, 1e-9)
    # feeder status is observed in real time: assume the CURRENT limit persists over the horizon (no look-ahead)
    Gh = np.full(H, C.Gh[i]) if C.Gh is not None else None
    d = dict(E=C.E, Pb=C.Pb, N=C.N, G=C.G, Gh=Gh, dem_dc=dem_dc[None], dem_sw=dem_sw[None], ub_dc=sv[None],
             ub_sw=swap_served(dem_sw, C.Csw, C.wsw, C.beta)[None], psw_cap=C.psw_cap, re=re[None], P=P[None],
             Tar=C.tar[hrs], wear=C.wear_fc[idx], pn_dc=C.pdc_ref[i] * day['m_dc'][sl] * (1 - VAR_OPEX) - day['cred'][sl] * C.pdc_ref[i] * rr,
             pn_sw=C.psw_ref[i] * day['m_sw'][sl] * (1 - VAR_OPEX), pi=pol['pi'], sw_var=C.sw_var, cal=PACK_CAL_RS_MWH_H,
             sell=1.0 if pol['sell_re'] else 0.0, init=(st['soc'], st['F']), q_fix=q, z_fix=z)
    Lbar = ctx['Lbar']
    d['nu_soc'] = 0.80 * Lbar
    d['nu_F'] = 0.80 * Lbar / (hw.eta_sw * (1 - DIST_LOSS))
    r = L.solve(L.struct(H, 1, hw, False), d, hw)
    if r is None:
        return None
    s0 = r['sc'][0]
    return dict(bc=s0['bc'][0], bd=s0['bd'][0], psw=s0['psw'][0])


def rule_plan(C, st, h, hw):
    """incumbent rules: ToU battery schedule; swap racks charged as soon as energy is handed out (to 100 %)"""
    bc = bd = 0.0
    if 11 <= h <= 15 or 1 <= h <= 5:
        bc = 0.5 * C.Pb
    if 18 <= h <= 21:
        bd = 0.6 * C.Pb
    room = (FULL * C.N - st['F'] * (1 - PACK_SD)) / hw.eta_sw
    psw = max(0.0, min(C.psw_cap, room))
    return dict(bc=bc, bd=bd, psw=psw)


def dispatch(C, st, h, i, plan, q, z, Dd, Ds, hw, pol, Pdam, sig):
    """execute one hour against the realised demand / RE; physical priorities: customers first,
    then swap-rack charging, then BESS charging. Returns a dict of flows (q, z are what was actually delivered)."""
    ETA = hw.eta_dc
    Gt = C.Gh[i] if C.Gh is not None else C.G
    soc, F = st['soc'], st['F']
    Fres = RESERVE * FULL * C.N
    capdc = PEAK_FACTOR * CHARGER_AVAIL * C.Pdc * derate(C.T[i])
    sdc, red0 = hub_served(Dd, capdc, C.hubw, sig, ret_red=True)
    sdc, red0 = float(sdc), float(red0)
    sdc0 = sdc
    ssw = max(0.0, min(float(swap_served(Ds, C.Csw, C.wsw, C.beta)), F - Fres))
    sched = (1 - TRANS_LOSS) * q + z
    if sched > Gt:                                      # scheduled energy the connection cannot carry is not delivered
        k = Gt / sched
        q, z, sched = q * k, z * k, Gt
    reg = min(C.re[i], max(0.0, Gt - sched))            # RE is wheeled through the same connection
    socmin, socmax = BESS_SOC_MIN * C.E, BESS_SOC_MAX * C.E
    s_ = soc * (1 - BESS_SD)
    bd = min(plan['bd'], C.Pb, max(0.0, (s_ - socmin) * BESS_ETA))
    bc = min(plan['bc'], C.Pb, max(0.0, (socmax - s_) / BESS_ETA)) if bd <= 1e-9 else 0.0
    Fn0 = F * (1 - PACK_SD) - ssw
    psw = max(0.0, min(plan['psw'], C.psw_cap, (FULL * C.N - Fn0) / hw.eta_sw))
    tar_c = C.tar[h % 24]
    cost_v = UNSCHED_PREM * tar_c
    cost_bd = C.wear_t[i] + 0.8 * (Pdam + DAM_ADDER * 1000)
    ld = 1 - DIST_LOSS

    def load():
        return (sdc / ETA + psw) / ld + bc

    v = u = curt = 0.0
    avail = sched + reg + bd
    L0 = load()
    if avail >= L0:
        sur = avail - L0
        if bd <= 1e-9:
            add = min(sur, C.Pb - bc, max(0.0, (socmax - s_) / BESS_ETA - bc))
            bc += max(add, 0.0); sur -= max(add, 0.0)
        room = max(0.0, (FULL * C.N - Fn0) / hw.eta_sw - psw)
        add = min(sur * ld, C.psw_cap - psw, room)
        if add > 0:
            psw += add; sur -= add / ld
        if pol['sell_re']:
            u = sur
        else:
            curt = min(sur, reg)
            u = sur - curt
    else:
        short = L0 - avail
        x = min(short, bc); bc -= x; short -= x
        head = min(STANDBY * Gt, max(0.0, Gt - (sched + reg)))
        extra_bd_cap = min(C.Pb - bd, max(0.0, (s_ - socmin) * BESS_ETA - bd)) if bc <= 1e-9 else 0.0
        for src in (['v', 'bd'] if cost_v <= cost_bd else ['bd', 'v']):
            if short <= 1e-9:
                break
            if src == 'v':
                x = min(short, head); v += x; head -= x; short -= x
            else:
                x = min(short, extra_bd_cap); bd += x; extra_bd_cap -= x; short -= x
        if short > 1e-9:
            x = min(short * ld, psw); psw -= x; short -= x / ld
        if short > 1e-9:
            cut = min(sdc, short * ld * ETA); sdc -= cut; short -= cut / (ETA * ld)
    soc_n = s_ + BESS_ETA * bc - bd / BESS_ETA
    F_n = F * (1 - PACK_SD) + hw.eta_sw * psw - ssw
    r_used = reg - curt
    curt = C.re[i] - r_used
    gross = r_used + q + z + v
    L_chg = sdc * (1 / ETA - 1) + psw * (1 - hw.eta_sw)
    L_dist = (sdc / ETA + psw) * DIST_LOSS / ld + TRANS_LOSS * q
    L_bess = bc * (1 - BESS_ETA) + bd * (1 / BESS_ETA - 1) + BESS_SD * soc
    L_idle = PACK_SD * F
    red = red0 * (sdc / sdc0 if sdc0 > 1e-9 else 0.0)
    return dict(q=q, z=z, red=red, sdc=sdc, ssw=ssw, bc=bc, bd=bd, psw=psw, v=v, u=u, curt=curt, r_used=r_used, gross=gross,
                L_chg=L_chg, L_dist=L_dist, L_bess=L_bess, L_idle=L_idle, soc=soc_n, F=F_n)


def simulate(cl, df, calib, pol, hw=None, fc=None, eps_hat=None, rho_hat=0.25, day0=0, ndays=365, seed=0,
             N=None, grid_scale=1.0, S=10, Lbar=None, eps_scale=1.0, sigma=None, beta=None):
    hw = hw or L.HW()
    C = Cluster(cl, df, calib, hw, N, grid_scale)
    C.sigmax = SIG_MAX if sigma is None else sigma
    C.beta = (BETA_REBAL if beta is None else beta) if pol.get('rebal') else 0.0
    if pol.get('rebal'):
        C.sw_var += PACK_REBAL_RS_MWH
    rng = np.random.default_rng(seed)
    i_start = len(df) - 8760 + day0 * 24                  # simulation year = the last 365 days of the data
    # starting price level = mean of the PREVIOUS 12 months (no look-ahead)
    ctx = dict(fc=fc, eps_hat=eps_hat if eps_hat is not None else (dg.EPS_DC, dg.EPS_SW), rho_hat=rho_hat, S=S,
               Lbar=Lbar if Lbar else float(C.dam[max(i_start - 8760, 0):i_start].mean() + DAM_ADDER * 1000))
    st = dict(soc=0.5 * C.E, F=0.6 * FULL * C.N, prev_delta=dict(dc=np.zeros(4), sw=np.zeros(4)))
    rt_state = dict(e_dc=1.0, e_sw=1.0, d_re=0.0)
    rows = []
    for dd in range(ndays):
        i0 = i_start + dd * 24
        day = da_plan(C, st, i0, pol, rng, hw, ctx)
        m_dc, m_sw = day['m_dc'], day['m_sw']
        hrs = np.arange(24)
        Dd_all, delta_dc = resp(C.dc_ref[i0:i0 + 24], m_dc[:24], hrs, dg.EPS_DC * eps_scale, RHO_TRUE, st['prev_delta']['dc'])
        Ds_all, delta_sw = resp(C.sw_ref[i0:i0 + 24], m_sw[:24], hrs, dg.EPS_SW * eps_scale, RHO_TRUE, st['prev_delta']['sw'])
        for h in range(24):
            i = i0 + h
            if pol['rt'] == 'mpc' and day['plan'] is not None:
                plan = rt_mpc(C, st, h, day, i, pol, hw, ctx, rt_state) or rule_plan(C, st, h, hw)
            else:
                plan = rule_plan(C, st, h, hw)
            Pd = C.dam[i]
            r = dispatch(C, st, h, i, plan, day['q'][h], day['z'][h], Dd_all[h], Ds_all[h], hw, pol, Pd, day['sig'][h])
            q, z = r['q'], r['z']                           # what was actually delivered / billed
            pdc, psw_ = C.pdc_ref[i] * m_dc[h], C.psw_ref[i] * m_sw[h]
            rev_dc, rev_sw = pdc * r['sdc'], psw_ * r['ssw']
            e_re = RE_PPA * 1000 * C.re[i]
            e_dam = q * (Pd + DAM_ADDER * 1000)
            e_z = z * C.tar[h]
            e_v = r['v'] * UNSCHED_PREM * C.tar[h]
            e_u = -r['u'] * SELL_FACTOR * Pd
            c_steer = day['cred'][h] * C.pdc_ref[i] * r['red']
            var = VAR_OPEX * (rev_dc + rev_sw) + C.sw_var * r['ssw'] + c_steer
            wear = C.wear_t[i] * r['bd'] + PACK_CAL_RS_MWH_H * st['F']
            rows.append((i, Dd_all[h], Ds_all[h], C.dc_ref[i], C.sw_ref[i], r['sdc'], r['ssw'], q, z, r['v'], r['u'], r['curt'],
                         C.re[i], r['r_used'], r['bc'], r['bd'], r['soc'], r['F'], rev_dc, rev_sw, e_re, e_dam, e_z, e_v, e_u,
                         var, wear, r['gross'], r['L_chg'], r['L_dist'], r['L_bess'], r['L_idle'], m_dc[h], m_sw[h],
                         r['sdc'] + r['ssw'], Dd_all[h] + Ds_all[h], c_steer, r['red']))
            fdc, fsw = day['dd'][h], day['ds'][h]
            rt_state['e_dc'] = float(np.clip(Dd_all[h] / max(fdc, 1e-6), 0.5, 1.6))
            rt_state['e_sw'] = float(np.clip(Ds_all[h] / max(fsw, 1e-6), 0.5, 1.6))
            rt_state['d_re'] = float(C.re[i] - day['f']['re'][h])
            st['soc'], st['F'] = r['soc'], r['F']
        st['prev_delta']['dc'] = delta_dc[-4:].copy(); st['prev_delta']['sw'] = delta_sw[-4:].copy()
    cols = ['i', 'dem_dc', 'dem_sw', 'ref_dc', 'ref_sw', 'sdc', 'ssw', 'q', 'z', 'v', 'u', 'curt', 're_gen', 're_used', 'bc', 'bd', 'soc',
            'F', 'rev_dc', 'rev_sw', 'c_re', 'c_dam', 'c_disc', 'c_unsched', 'c_sell', 'c_var', 'c_wear', 'gross', 'L_chg',
            'L_dist', 'L_bess', 'L_idle', 'm_dc', 'm_sw', 'served', 'demand', 'c_steer', 'redirected']
    R = pd.DataFrame(rows, columns=cols)
    R['hour'] = C.hour[R['i'].values]
    R['month'] = df.index.month.values[R['i'].values]
    R['T'] = C.T[R['i'].values]
    R['dam'] = C.dam[R['i'].values]
    R.attrs['cap'] = C.c['cap']
    return R


def kpis(R, fixed_cr=0.0):
    """annual KPIs for one cluster result frame (c_var already includes the steering credit)"""
    n = len(R)
    rev = R.rev_dc.sum() + R.rev_sw.sum()
    fixed = fixed_cr * 1e7 * n / 8760.0
    cost = R.c_re.sum() + R.c_dam.sum() + R.c_disc.sum() + R.c_unsched.sum() + R.c_sell.sum() + R.c_var.sum() + R.c_wear.sum() + fixed
    ebitda = rev - cost
    stored = (R.soc.iloc[-1] - R.soc.iloc[0]) + (R.F.iloc[-1] - R.F.iloc[0])
    loss = R.L_chg.sum() + R.L_dist.sum() + R.L_bess.sum() + R.L_idle.sum()
    consumed = R.gross.sum() - R.u.sum() - stored
    return dict(
        reliability=R.served.sum() / R.demand.sum(),
        rel_dc=R.sdc.sum() / R.dem_dc.sum(), rel_sw=R.ssw.sum() / R.dem_sw.sum(),
        loss_pct=loss / consumed, loss_chg=R.L_chg.sum() / consumed, loss_dist=R.L_dist.sum() / consumed,
        loss_bess=R.L_bess.sum() / consumed, loss_idle=R.L_idle.sum() / consumed,
        revenue_cr=rev / 1e7, ebitda_cr=ebitda / 1e7, margin=ebitda / rev,
        energy_gwh=R.served.sum() / 1e3, consumed_gwh=consumed / 1e3, re_gen_gwh=R.re_gen.sum() / 1e3,
        demand_gwh=R.demand.sum() / 1e3, vol_vs_ref=R.served.sum() / (R.ref_dc.sum() + R.ref_sw.sum()),
        re_util=R.re_used.sum() / R.re_gen.sum(), re_sold=(R.u.sum()) / 1e3,
        e_cost_kwh=(R.c_re.sum() + R.c_dam.sum() + R.c_disc.sum() + R.c_unsched.sum() + R.c_sell.sum()) / R.served.sum() / 1000.0,
        unsched_pct=R.v.sum() / R.gross.sum(), swap_stockout=1 - R.ssw.sum() / R.dem_sw.sum(),
        bess_gwh=R.bd.sum() / 1e3, steer_cr=R.c_steer.sum() / 1e7, dam_share=R.q.sum() / R.gross.sum())
