"""Side-by-side comparison of the SYNTHETIC-data and REAL-data (IEX + ERA5) runs."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results') + os.sep
BLUE, ORANGE, GRAY, INK, INK2, GRID = '#2a78d6', '#eb6834', '#8a8984', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'axes.axisbelow': True,
                     'axes.titleweight': 'bold', 'figure.dpi': 150, 'axes.edgecolor': GRID})
L = {m: pd.read_csv(R + f'{m}/tables/ladder_network_full.csv', index_col=0) for m in ('synthetic', 'real')}
tbl = pd.concat({m: L[m][['reliability', 'loss', 'margin', 'min_margin', 'ebitda_cr', 're_util']] for m in L}, axis=1)
tbl.to_csv(R + 'comparison_ladder.csv'); print(tbl.round(3).to_string())
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
for a, (k, ttl, tgt, tl) in zip(ax, [('reliability', 'Service reliability', 0.897, 'target 89.7%'), ('loss', 'Losses', 0.08, 'target 8%'), ('margin', 'EBITDA margin', 0.12, 'floor 12%')]):
    x = np.arange(len(L['real'])); w = 0.38
    a.bar(x - w / 2, L['synthetic'][k], w, color=GRAY, label='synthetic weather + price')
    a.bar(x + w / 2, L['real'][k], w, color=BLUE, label='real IEX + ERA5 weather')
    a.axhline(tgt, color=ORANGE, ls='--', lw=1.3); a.set_title(f'{ttl} (dashed = {tl})', fontsize=10)
    a.set_xticks(x); a.set_xticklabels([t.split(' ')[0] for t in L['real'].index]); a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0 if k == 'reliability' else 1))
    if k == 'reliability': a.set_ylim(0.6, 1.0)
ax[0].legend(frameon=False, fontsize=8, loc='upper left'); fig.tight_layout(); fig.savefig(R + 'fig_compare.png', bbox_inches='tight', facecolor='white')
