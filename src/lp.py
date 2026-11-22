"""Generic energy-network LP used by BOTH the day-ahead stochastic planner (S scenarios, H=30 h) and the
real-time receding-horizon controller (S=1, H<=6 h).  First stage: DAM bid q_h, scheduled DISCOM z_h.
Recourse (per scenario): served DC / swap energy, swap-rack charging, BESS charge/discharge, RE use,
unscheduled DISCOM draw v, surplus sale u, and the state trajectories (BESS SoC, rack energy F).
Swap stations exchange packs 1-for-1 (an empty pack goes straight into the rack and recharges), so the number of
packs in the racks is constant; rack energy F falls by the energy handed out and rises with charging.
Units: MWh (hourly), prices Rs/MWh."""
import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
from params import *

_C = {}
FULL = PACK_FULL - PACK_EMPTY           # usable fraction of a pack (0.85)
NAMES = ['sdc', 'ssw', 'psw', 'bc', 'bd', 'r', 'v', 'u']


class HW:
    def __init__(self, eta_dc=ETA_DC, eta_sw=ETA_SW):
        self.eta_dc, self.eta_sw = eta_dc, eta_sw


def struct(H, S, hw, cvar):
    key = (H, S, hw.eta_dc, hw.eta_sw, cvar)
    if key in _C:
        return _C[key]
    off = {}; n = 0
    off['q'] = n; n += H
    off['z'] = n; n += H
    sc = []
    for w in range(S):
        o = {}
        for nm in NAMES:
            o[nm] = n; n += H
        for nm in ['soc', 'F']:
            o[nm] = n; n += H + 1
        sc.append(o)
    if cvar:
        off['t'] = n; n += 1
        off['w'] = n; n += S
    nvar = n
    er, ec, ev, ur, uc, uv = [], [], [], [], [], []
    for w in range(S):
        o = sc[w]
        for h in range(H):
            r0 = w * 3 * H + 0 * H + h      # energy balance
            for col, val in [(o['r'] + h, 1), (off['q'] + h, 1 - TRANS_LOSS), (off['z'] + h, 1), (o['v'] + h, 1), (o['bd'] + h, 1),
                             (o['sdc'] + h, -1 / (hw.eta_dc * (1 - DIST_LOSS))), (o['psw'] + h, -1 / (1 - DIST_LOSS)),
                             (o['bc'] + h, -1), (o['u'] + h, -1)]:
                er.append(r0); ec.append(col); ev.append(val)
            r1 = w * 3 * H + 1 * H + h      # BESS state
            for col, val in [(o['soc'] + h + 1, 1), (o['soc'] + h, -(1 - BESS_SD)), (o['bc'] + h, -BESS_ETA), (o['bd'] + h, 1 / BESS_ETA)]:
                er.append(r1); ec.append(col); ev.append(val)
            r2 = w * 3 * H + 2 * H + h      # rack energy
            for col, val in [(o['F'] + h + 1, 1), (o['F'] + h, -(1 - PACK_SD)), (o['psw'] + h, -hw.eta_sw), (o['ssw'] + h, 1)]:
                er.append(r2); ec.append(col); ev.append(val)
            u0 = w * 2 * H + 0 * H + h      # grid connection
            for col, val in [(o['r'] + h, 1), (off['q'] + h, 1 - TRANS_LOSS), (off['z'] + h, 1), (o['v'] + h, 1)]:
                ur.append(u0); uc.append(col); uv.append(val)
            u1 = w * 2 * H + 1 * H + h      # handed-out energy must come from racked energy above the reserve
            for col, val in [(o['ssw'] + h, 1), (o['F'] + h, -1)]:
                ur.append(u1); uc.append(col); uv.append(val)
    neq, nub = S * 3 * H, S * 2 * H
    A_eq = sp.csr_matrix((ev, (er, ec)), shape=(neq, nvar))
    A_ub = sp.csr_matrix((uv, (ur, uc)), shape=(nub, nvar))
    st = dict(H=H, S=S, off=off, sc=sc, nvar=nvar, A_eq=A_eq, A_ub=A_ub, neq=neq, nub=nub, cvar=cvar)
    _C[key] = st
    return st


def solve(st, d, hw):
    """d: dict of data (keys used below). returns dict or None"""
    H, S, off, sc, nvar = st['H'], st['S'], st['off'], st['sc'], st['nvar']
    E, Pb, N, G = d['E'], d['Pb'], d['N'], d['G']
    Fres = RESERVE * FULL * N
    lb = np.zeros(nvar); ub = np.full(nvar, 1e9)
    qmax = d.get('qmax', G / (1 - TRANS_LOSS))
    if d.get('q_fix') is not None:
        lb[off['q']:off['q'] + H] = ub[off['q']:off['q'] + H] = d['q_fix']
        lb[off['z']:off['z'] + H] = ub[off['z']:off['z'] + H] = d['z_fix']
    else:
        ub[off['q']:off['q'] + H] = qmax
        ub[off['z']:off['z'] + H] = G
    pc = np.zeros((S, nvar))
    const = np.zeros(S)
    beq = np.zeros(st['neq']); bub = np.zeros(st['nub'])
    soc0, F0 = d['init']
    for w in range(S):
        o = sc[w]
        ub[o['sdc']:o['sdc'] + H] = d['ub_dc'][w]
        ub[o['ssw']:o['ssw'] + H] = np.minimum(d['dem_sw'][w], d['ub_sw'][w])
        ub[o['psw']:o['psw'] + H] = d['psw_cap']
        ub[o['bc']:o['bc'] + H] = Pb
        ub[o['bd']:o['bd'] + H] = Pb
        ub[o['r']:o['r'] + H] = d['re'][w]
        ub[o['v']:o['v'] + H] = STANDBY * G
        lb[o['soc']:o['soc'] + H + 1] = BESS_SOC_MIN * E; ub[o['soc']:o['soc'] + H + 1] = BESS_SOC_MAX * E
        lb[o['F']:o['F'] + H + 1] = min(Fres, F0); ub[o['F']:o['F'] + H + 1] = FULL * N
        for nm, v0 in (('soc', soc0), ('F', F0)):
            lb[o[nm]] = ub[o[nm]] = v0
        bub[w * 2 * H + 0 * H: w * 2 * H + 1 * H] = d['Gh'] if d.get('Gh') is not None else G
        bub[w * 2 * H + 1 * H: w * 2 * H + 2 * H] = -Fres
        P = d['P'][w]
        pc[w, o['sdc']:o['sdc'] + H] = d['pn_dc'] + d['pi']
        pc[w, o['ssw']:o['ssw'] + H] = d['pn_sw'] + d['pi'] - d['sw_var']
        pc[w, off['q']:off['q'] + H] = -(P + DAM_ADDER * 1000)
        pc[w, off['z']:off['z'] + H] = -d['Tar']
        pc[w, o['v']:o['v'] + H] = -UNSCHED_PREM * d['Tar']
        pc[w, o['u']:o['u'] + H] = d['sell'] * SELL_FACTOR * P
        pc[w, o['bd']:o['bd'] + H] = -d['wear']
        pc[w, o['F'] + 1:o['F'] + H + 1] = -d['cal']
        pc[w, o['soc'] + H] += d['nu_soc']
        pc[w, o['F'] + H] += d['nu_F']
        pc[w, o['psw']:o['psw'] + H] += -1e-3          # tie-break
        const[w] = -RE_PPA * 1000.0 * d['re'][w].sum()  # take-or-pay renewable cost: scenario-dependent, sunk w.r.t. decisions
    if st['cvar']:
        kap, alp = d['kappa'], d['alpha']
        c = -(1 - kap) * pc.mean(axis=0)
        c[off['t']] = -kap
        c[off['w']:off['w'] + S] = kap / ((1 - alp) * S)
        lb[off['t']] = -1e12; ub[off['t']] = 1e12
        cvr = sp.lil_matrix((S, nvar))
        for w in range(S):
            cvr[w, :] = -pc[w]
            cvr[w, off['t']] = 1
            cvr[w, off['w'] + w] = -1
        A_ub = sp.vstack([st['A_ub'], cvr.tocsr()]).tocsr()
        bub = np.concatenate([bub, const])              # t - w_s <= pc_s.x + const_s
    else:
        c = -pc.mean(axis=0)
        A_ub = st['A_ub']
    res = linprog(c, A_ub=A_ub, b_ub=bub, A_eq=st['A_eq'], b_eq=beq, bounds=np.column_stack([lb, ub]), method='highs')
    if res.status != 0:
        return None
    x = res.x
    out = dict(obj=-res.fun, q=x[off['q']:off['q'] + H].copy(), z=x[off['z']:off['z'] + H].copy(), sc=[])
    for w in range(S):
        o = sc[w]
        out['sc'].append({nm: x[o[nm]:o[nm] + H].copy() for nm in NAMES})
        out['sc'][-1]['soc'] = x[o['soc']:o['soc'] + H + 1].copy()
        out['sc'][-1]['F'] = x[o['F']:o['F'] + H + 1].copy()
    return out
