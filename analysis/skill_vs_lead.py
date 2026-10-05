"""
Domain skill at T=6, ACC and RMSE vs lead and RZSM contribution.

    python analysis/skill_vs_lead.py --runs runs --outdir figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from common import (COLORS, CONFIGS, DOMAIN_RUNS, LEAD_DAYS, LEAD_FIGURE_RUNS, LEAD_FIGURE_SEEDS,
                    LEAD_LABELS, MARKERS, load_runs, save, seed_stats)

LINES = {'NIRv+RZSM': dict(ls='-', lw=2.2, ms=7, zorder=4),
         'NIRv+lagRZSM': dict(ls='--', lw=1.8, ms=6, zorder=3),
         'NIRv-only': dict(ls='-.', lw=1.8, ms=6, zorder=2)}

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'legend.fontsize': 9, 'axes.spines.top': False,
    'axes.spines.right': False,
})


def style(config):
    return dict(color=COLORS[config], marker=MARKERS[config], linestyle=LINES[config]['ls'],
                linewidth=LINES[config]['lw'], markersize=LINES[config]['ms'],
                zorder=LINES[config]['zorder'])


def lead_axis(ax, ylabel):
    ax.set_xlabel('Lead time (days)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(LEAD_DAYS)
    ax.set_xticklabels(LEAD_LABELS)
    ax.set_xlim(LEAD_DAYS[0] - 4, LEAD_DAYS[-1] + 4)


def domain_skill_table(runs) -> pd.DataFrame:
    rows = []
    for config in CONFIGS:
        row = {'Model': config}
        for key, label in (('acc', 'ACC'), ('rmse', 'RMSE'), ('r2', 'R2'), ('skill_score', 'SS')):
            mean, std = seed_stats(runs[config], key)
            row[label] = f'{mean:.3f} +/- {std:.3f}'
        rows.append(row)
    return pd.DataFrame(rows)


def plot_skill_vs_lead(runs, outdir):
    acc = {c: seed_stats(runs[c], 'acc_lead')[0] for c in CONFIGS}
    rmse = {c: seed_stats(runs[c], 'rmse_lead')[0] for c in CONFIGS}
    clim = seed_stats(runs['NIRv+RZSM'], 'clim_rmse_lead')[0]

    fig, (ax_acc, ax_rmse) = plt.subplots(1, 2, figsize=(10, 4.5))
    for config in CONFIGS:
        ax_acc.plot(LEAD_DAYS, acc[config], label=config, **style(config))
    lead_axis(ax_acc, 'ACC')
    values = np.concatenate(list(acc.values()))
    ax_acc.set_ylim(max(0, np.floor(values.min() * 10) / 10 - 0.05),
                    min(1, np.ceil(values.max() * 10) / 10 + 0.05))
    ax_acc.legend(loc='upper right', framealpha=0.9, edgecolor='0.8')

    ax_rmse.plot(LEAD_DAYS, clim, color='#636363', linestyle=':', linewidth=1.8, marker='D',
                 markersize=5, label='NIRv Clim', zorder=1)
    for config in CONFIGS:
        ax_rmse.plot(LEAD_DAYS, rmse[config], label=config, **style(config))
    lead_axis(ax_rmse, 'RMSE')
    values = np.concatenate(list(rmse.values()))
    ax_rmse.set_ylim(np.floor(values.min() * 10) / 10 - 0.05,
                     max(np.ceil(values.max() * 10) / 10, np.ceil(clim.max() * 10) / 10) + 0.05)
    ax_rmse.legend(loc='upper right', framealpha=0.9, edgecolor='0.8')

    for ax, letter in ((ax_acc, 'A'), (ax_rmse, 'B')):
        ax.text(-0.12, 1.05, letter, transform=ax.transAxes, fontsize=14, fontweight='bold',
                va='bottom', ha='right')
    fig.tight_layout()
    save(fig, outdir, 'skill_vs_lead')


def plot_rzsm_contribution(runs, outdir):
    acc = {c: seed_stats(runs[c], 'acc_lead')[0] for c in CONFIGS}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for other, ls in (('NIRv+lagRZSM', ':'), ('NIRv-only', '--')):
        ax.plot(LEAD_DAYS, acc['NIRv+RZSM'] - acc[other], color=COLORS[other],
                marker=MARKERS[other], linestyle=ls, linewidth=2, markersize=6.5, zorder=3,
                label=rf'$\Delta$ACC (NIRv+RZSM $-$ {other})')
    lead_axis(ax, r'$\Delta$ACC')
    deltas = np.concatenate([acc['NIRv+RZSM'] - acc[o] for o in ('NIRv+lagRZSM', 'NIRv-only')])
    ax.set_ylim(min(-0.005, deltas.min() - 0.005), deltas.max() + 0.015)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='0.8')
    fig.tight_layout()
    save(fig, outdir, 'rzsm_contribution')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='runs')
    p.add_argument('--outdir', default='figures')
    p.add_argument('--tlen', type=int, default=6)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    table = domain_skill_table({c: load_runs(args.runs, DOMAIN_RUNS[c], args.tlen) for c in CONFIGS})
    print(table.to_string(index=False))
    table.to_csv(outdir / 'domain_skill.csv', index=False)

    runs = {c: load_runs(args.runs, LEAD_FIGURE_RUNS[c], args.tlen, LEAD_FIGURE_SEEDS) for c in CONFIGS}
    plot_skill_vs_lead(runs, outdir)
    plot_rzsm_contribution(runs, outdir)


if __name__ == '__main__':
    main()
