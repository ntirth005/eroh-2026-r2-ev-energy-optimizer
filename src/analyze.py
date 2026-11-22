"""Builds all result tables and charts for the write-up from the simulation outputs."""
import glob, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from params import *

OUT = '../out/charts/'
TB = '../out/tables/'
BLUE, ORANGE, AQUA, MAG, GRAY, INK, INK2, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#e87ba4', '#8a8984', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150, 'axes.titleweight': 'bold',
                     'axes.titlesize': 11, 'axes.titlecolor': INK, 'text.color': INK, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.axisbelow': True})
cal = json.load(open('../data/calib.json'))


def agg(g):
    """network aggregation: reliability by demand, losses by energy consumed, renewable use by renewable generated"""
    return pd.Series(dict(reliability=g.energy_gwh.sum() / g.demand_gwh.sum(), margin=g.ebitda_cr.sum() / g.revenue_cr.sum(),
                          revenue_cr=g.revenue_cr.sum(), ebitda_cr=g.ebitda_cr.sum(), min_margin=g.margin.min(), min_rel=g.reliability.min(),
                          loss=np.average(g.loss_pct, weights=g.consumed_gwh), energy_gwh=g.energy_gwh.sum(),
                          re_util=np.average(g.re_util, weights=g.re_gen_gwh), rel_sw=1 - np.average(g.swap_stockout, weights=g.demand_gwh),
                          bess_gwh=g.bess_gwh.sum(), steer_cr=g.steer_cr.sum(), dam_share=np.average(g.dam_share, weights=g.consumed_gwh)))


def save(fig, name):
    fig.savefig(OUT + name, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    raw = pd.read_csv(TB + 'all_runs_raw.csv')
    lad = raw[~raw.tag.str.startswith(('sigma_', 'SC|', 'SE|'))].copy()
    lad.to_csv(TB + 'ladder_by_cluster.csv', index=False)
    net = lad.groupby('tag', sort=False).apply(agg)
    net.to_csv(TB + 'ladder_network_full.csv')
    short = {t: t.split(' ')[0] for t in net.index}
    # ---------------- fig1: the ladder
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    labs = [t.split(' ', 1)[0] + '\n' + t.split(' ', 1)[1].replace('+', '').replace(' ', '\n', 1) for t in net.index]
    cols = [GRAY] + [BLUE] * (len(net) - 1)
    for a, (k, ttl, tgt, tl, fmt) in zip(ax, [('reliability', 'Service reliability', 0.897, 'target 89.7%', '{:.1%}'),
                                              ('loss', 'Storage & conversion losses', 0.08, 'target 8%', '{:.1%}'),
                                              ('margin', 'EBITDA margin (network)', 0.12, 'floor 12%', '{:.1%}')]):
        v = net[k].values
        a.bar(range(len(v)), v, color=cols, width=0.62)
        a.axhline(tgt, color=ORANGE, lw=1.4, ls='--')
        for i, x in enumerate(v):
            a.text(i, x, fmt.format(x), ha='center', va='bottom', fontsize=8, color=INK)
        a.set_xticks(range(len(v))); a.set_xticklabels([t.split(' ')[0] for t in net.index], fontsize=9)
        a.set_title(f'{ttl}\n(dashed line = {tl})', fontsize=10); a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0 if k == 'reliability' else 1))
        lo = 0.6 if k == 'reliability' else 0
        a.set_ylim(lo, max(v.max(), tgt) * 1.10)
        if k != 'reliability': a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    fig.text(0.5, -0.04, 'A0 incumbent · A1 +ML forecasts · A2 +deterministic MPC · A3 +stochastic CVaR · A4 +dynamic pricing · A5 +hub steering · A6 +RE sales & pack rebalancing · A7 +SiC chargers',
             ha='center', fontsize=8, color=INK2)
    fig.tight_layout(); save(fig, 'fig1_ladder.png')
    # ---------------- fig2: cluster view
    a0 = lad[lad.tag.str.startswith('A0')].set_index('cluster').loc[SHORT]
    a6 = lad[lad.tag.str.startswith('A7')].set_index('cluster').loc[SHORT]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(5); w = 0.36
    for a, k, ttl, ref in ((ax[0], 'reliability', 'Reliability by cluster', 0.897), (ax[1], 'margin', 'EBITDA margin by cluster', 0.12)):
        a.bar(x - w / 2, a0[k], w, color=GRAY, label='A0 incumbent')
        a.bar(x + w / 2, a6[k], w, color=BLUE, label='A7 full AI framework')
        for i in range(5):
            a.text(i - w / 2, a0[k].iloc[i], f'{a0[k].iloc[i]:.0%}', ha='center', va='bottom', fontsize=8)
            a.text(i + w / 2, a6[k].iloc[i], f'{a6[k].iloc[i]:.0%}', ha='center', va='bottom', fontsize=8)
        a.axhline(ref, color=ORANGE, ls='--', lw=1.3)
        a.set_xticks(x); a.set_xticklabels([NAME[s] for s in SHORT], fontsize=9); a.set_title(ttl)
        a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
        if k == 'reliability': a.set_ylim(0.6, 1.0)
    ax[0].legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout(); save(fig, 'fig2_clusters.png')
    # ---------------- loss decomposition
    rows = []
    for t in net.index:
        g = lad[lad.tag == t]; wts = g.consumed_gwh
        rows.append((t.split(' ')[0], *[np.average(g[c], weights=wts) for c in ('loss_chg', 'loss_dist', 'loss_bess', 'loss_idle')]))
    L = pd.DataFrame(rows, columns=['policy', 'Charger conversion', 'Distribution & aux', 'BESS round-trip', 'Idle-pack self-discharge'])
    L.to_csv(TB + 'loss_decomposition.csv', index=False)
    fig, a = plt.subplots(figsize=(7.5, 3.8))
    bot = np.zeros(len(L))
    for c, col in zip(L.columns[1:], (BLUE, ORANGE, AQUA, MAG)):
        a.bar(L.policy, L[c], bottom=bot, color=col, width=0.62, label=c, edgecolor='white', linewidth=1.2)
        bot += L[c].values
    for i, v in enumerate(bot):
        a.text(i, v, f'{v:.1%}', ha='center', va='bottom', fontsize=8)
    a.axhline(0.08, color=INK, ls='--', lw=1); a.text(-0.45, 0.0815, 'target 8%', ha='left', fontsize=8, color=INK)
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0)); a.set_title('Where the losses are (share of energy consumed)')
    a.legend(frameon=False, fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); save(fig, 'fig3_losses.png')
    # ---------------- forecast accuracy
    fa = pd.read_csv(TB + 'forecast_accuracy.csv')
    fig, a = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(fa))[::-1]
    a.barh(y + 0.19, fa.naive_wape, 0.36, color=GRAY, label='Seasonal-naive')
    best = fa.rec_wape.fillna(fa.lgbm_wape)
    a.barh(y - 0.19, best, 0.36, color=BLUE, label='LightGBM (+ hierarchical reconciliation for demand)')
    a.set_yticks(y); a.set_yticklabels(fa.series, fontsize=8)
    for yi, n_, b_ in zip(y, fa.naive_wape, best):
        a.text(max(n_, b_) + 0.005, yi, f'{b_ / n_ - 1:+.0%}', va='center', fontsize=8, color=INK2)
    a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0)); a.set_title('Out-of-sample forecast error on the test year (WAPE, day-ahead) – lower is better')
    a.legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout(); save(fig, 'fig4_forecast.png')
    # ---------------- sigma sweep
    sg = raw[raw.tag.str.startswith('sigma_')].copy(); sg['sigma'] = sg.tag.str.split('_').str[1].astype(float)
    sgn = sg.groupby('sigma').apply(agg).reset_index()
    sgn.to_csv(TB + 'steering_sweep.csv', index=False)
    fig, a = plt.subplots(1, 2, figsize=(10, 3.6))
    a[0].plot(sgn.sigma, sgn.reliability, color=BLUE, lw=2, marker='o', ms=5)
    a[0].axhline(0.897, color=ORANGE, ls='--'); a[0].text(0.0, 0.9, 'target 89.7%', color=ORANGE, fontsize=8, va='bottom')
    a[0].set_title('Reliability vs steering adoption'); a[0].set_xlabel('Max share of hot-hub overflow redirectable (sigma_max)')
    a[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
    a[1].plot(sgn.sigma, sgn.margin, color=BLUE, lw=2, marker='o', ms=5)
    a[1].axhline(0.12, color=ORANGE, ls='--'); a[1].set_title('EBITDA margin vs steering adoption'); a[1].set_xlabel('Steering adoption')
    a[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
    fig.tight_layout(); save(fig, 'fig5_steering.png')
    # ---------------- stress scenarios
    sc = raw[raw.tag.str.startswith('SC|')].copy()
    sc[['_', 'scenario', 'policy']] = sc.tag.str.split('|', expand=True)
    scn = sc.groupby(['scenario', 'policy'], sort=False).apply(agg).reset_index()
    scn.to_csv(TB + 'stress_scenarios.csv', index=False)
    order = list(dict.fromkeys(sc.scenario)); pols = list(dict.fromkeys(sc.policy))
    cmap = {'A0 Incumbent': GRAY, 'A2 Det. MPC': AQUA, 'A3 Stoch. MPC': ORANGE, 'A7 Full AI+SiC': BLUE}
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=False)
    for a, k, ttl in ((ax[0], 'reliability', 'Reliability in the stress window'), (ax[1], 'margin', 'EBITDA margin in the stress window')):
        wd = 0.2
        for j, p in enumerate(pols):
            v = [scn[(scn.scenario == s_) & (scn.policy == p)][k].iloc[0] for s_ in order]
            a.bar(np.arange(len(order)) + (j - 1.5) * wd, v, wd, color=cmap[p], label=p)
        a.set_xticks(range(len(order))); a.set_xticklabels(['Base\nwindow', 'Heatwave +5C\n& IEX x1.35', 'Wind lull\n+ cloud', 'IEX price\nspike x1.8', 'Feeder cut\n-50% eve', 'Demand\n+30%', 'Compound\nshock'], fontsize=8)
        a.set_title(ttl); a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
    ax[0].set_ylim(0.6, 1.0); ax[1].axhline(0.12, color=INK, ls='--', lw=1)
    h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, fontsize=9, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.06))
    ax[1].text(6.45, 0.122, '12% floor', fontsize=8, ha='right')
    fig.tight_layout(); save(fig, 'fig6_stress.png')
    # ---------------- sensitivities (full year)
    se = raw[raw.tag.str.startswith('SE|')].copy()
    se[['_', 'case', 'policy']] = se.tag.str.split('|', expand=True)
    sen = se.groupby(['case', 'policy'], sort=False).apply(agg).reset_index()
    base = {'A0 Incumbent': net.iloc[0], 'A7 Full AI+SiC': net.loc[[i for i in net.index if i.startswith('A7')][0]]}
    sen.to_csv(TB + 'sensitivities.csv', index=False)
    cs = list(dict.fromkeys(se.case))
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))
    for a, k, ttl in ((ax[0], 'reliability', 'Full-AI reliability under perturbation'), (ax[1], 'margin', 'Full-AI EBITDA margin under perturbation')):
        v = np.array([sen[(sen.case == c) & (sen.policy == 'A7 Full AI+SiC')][k].iloc[0] for c in cs])
        b0 = base['A7 Full AI+SiC'][k]
        o = np.argsort(-np.abs(v - b0))
        a.barh(np.arange(len(cs)), (v - b0)[o][::-1], color=[BLUE if x_ >= 0 else ORANGE for x_ in (v - b0)[o][::-1]], height=0.6)
        a.set_yticks(range(len(cs))); a.set_yticklabels(np.array(cs)[o][::-1], fontsize=8)
        a.axvline(0, color=INK, lw=0.8); a.set_title(ttl + f'\n(base {b0:.1%}; shown as change, pp)')
        a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x_, _: f'{x_ * 100:+.1f}'))
    fig.tight_layout(); save(fig, 'fig7_sensitivity.png')
    # ---------------- bootstrap risk on daily outcomes
    def daily(tag_glob):
        out = {}
        for s in SHORT:
            f = glob.glob(f'../data/runs/{tag_glob}__{s}.pkl')[0]
            R = pd.read_pickle(f)
            cost = R.c_re + R.c_dam + R.c_disc + R.c_unsched + R.c_sell + R.c_var + R.c_wear + cal[s]['fixed_cr'] * 1e7 / 8760
            rev = R.rev_dc + R.rev_sw
            d = pd.DataFrame(dict(rev=rev.values, e=(rev - cost).values, served=R.served.values, dem=R.demand.values)).groupby(np.arange(len(R)) // 24).sum()
            out[s] = d
        return out
    D0, D6 = daily('A0_Incumbent_rules'), daily('A7__SiC_charger_hardware')
    rng = np.random.default_rng(3); nb = 53; B = 7; res = []
    for it in range(4000):
        starts = rng.integers(0, 365, nb)
        idx = np.concatenate([(np.arange(s0, s0 + B)) % 365 for s0 in starts])[:365]       # circular blocks
        r = dict(it=it)
        for nm, D in (('A0', D0), ('A7', D6)):
            rv = sum(D[s].rev.values[idx].sum() for s in SHORT); eb = sum(D[s].e.values[idx].sum() for s in SHORT)
            r[nm + '_margin'] = eb / rv; r[nm + '_ebitda'] = eb / 1e7
            r[nm + '_rel'] = sum(D[s].served.values[idx].sum() for s in SHORT) / sum(D[s].dem.values[idx].sum() for s in SHORT)
            r[nm + '_minclu'] = min(D[s].e.values[idx].sum() / D[s].rev.values[idx].sum() for s in SHORT)
        res.append(r)
    R = pd.DataFrame(res)
    summ = pd.DataFrame({c: [R[c].quantile(q) for q in (0.05, 0.5, 0.95)] for c in R.columns if c != 'it'}, index=['P5', 'P50', 'P95']).T
    summ['P(min cluster margin>12%)'] = np.nan
    summ.loc['A7_minclu', 'P(min cluster margin>12%)'] = (R.A7_minclu > 0.12).mean()
    summ.loc['A0_minclu', 'P(min cluster margin>12%)'] = (R.A0_minclu > 0.12).mean()
    summ.to_csv(TB + 'bootstrap_risk.csv')
    fig, a = plt.subplots(figsize=(7.5, 3.6))
    a.hist(R.A0_margin, bins=40, color=GRAY, alpha=0.9, label='A0 incumbent'); a.hist(R.A7_margin, bins=40, color=BLUE, alpha=0.9, label='A7 full AI')
    a.axvline(0.12, color=ORANGE, ls='--'); a.text(0.121, a.get_ylim()[1] * 0.9, '12% floor', color=ORANGE, fontsize=8)
    a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0)); a.set_title('Network EBITDA margin – 4,000 resampled years (7-day block bootstrap)')
    a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); save(fig, 'fig8_bootstrap.png')
    # ---------------- operating-day view (Delhi, a hot-week day)
    A5 = pd.read_pickle(glob.glob('../data/runs/A7__SiC_charger_hardware__DEL.pkl')[0]); A0r = pd.read_pickle(glob.glob('../data/runs/A0_Incumbent_rules__DEL.pkl')[0])
    dmean = A5['T'].values.reshape(-1, 24).mean(axis=1); dday = int(np.argmax(dmean)); d0 = dday * 24
    h = np.arange(24)
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5))
    for R_, col, nm in ((A0r, GRAY, 'A0'), (A5, BLUE, 'A7')):
        s = R_.iloc[d0:d0 + 24]
        ax[0].plot(h, s.m_dc.values, color=col, lw=2, label=nm)
        ax[1].plot(h, s.sdc.values, color=col, lw=2)
        ax[2].plot(h, s.soc.values / CFG['DEL']['bess'], color=col, lw=2)
    ax[1].plot(h, A5.iloc[d0:d0 + 24].dem_dc.values, color=ORANGE, lw=1.3, ls='--'); ax[1].text(23, A5.iloc[d0 + 20].dem_dc, 'demand (AI prices)', color=ORANGE, fontsize=8, ha='right', va='bottom')
    ax[0].set_title('DC price multiplier vs market tariff'); ax[1].set_title('DC energy served, MWh/h'); ax[2].set_title('Stationary BESS state of charge')
    ax[0].axhline(1.2, color=GRID); ax[0].axhline(0.8, color=GRID); ax[0].legend(frameon=False, fontsize=8)
    ax[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0))
    for a in ax: a.set_xlabel('hour of day')
    fig.suptitle(f'Delhi NCR, hottest day of the test year (mean {dmean[dday]:.1f} C)', fontsize=10, y=1.02)
    fig.tight_layout(); save(fig, 'fig9_day.png')
    print(net.round(4).to_string()); print(summ.round(4).to_string())
    print(scn.round(3).to_string()); print(sen.round(3).to_string()); print(sgn.round(3).to_string())


if __name__ == '__main__':
    main()
