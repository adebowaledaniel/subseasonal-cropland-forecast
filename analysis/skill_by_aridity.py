"""
Skill by aridity regime: ACC vs lead (T=6), the RZSM delta-ACC (T=6 and T=10) and a table
of regime ACC per lead (T=6).

    python analysis/skill_by_aridity.py --runs runs --outdir figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from common import COLORS, LEAD_LABELS, LEADS, MARKERS, REGIME_RUNS, load_runs, save, seed_stats

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 10,
    'axes.labelsize': 11, 'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
})

ORDER = ['NIRv+RZSM', 'NIRv-only', 'NIRv+lagRZSM']
LINESTYLES = {'NIRv+RZSM': '-', 'NIRv-only': '--', 'NIRv+lagRZSM': ':'}
REGIMES = {'water_limited': 'WL', 'energy_limited': 'EL'}


def regime_acc(runs_root, tlen):
    """{config: {regime: seed-mean ACC per lead}}."""
    out = {}
    for config in ORDER:
        runs = load_runs(runs_root, REGIME_RUNS[config], tlen)
        out[config] = {r: seed_stats(runs, f'{r}_acc_lead')[0] for r in REGIMES}
    return out


def lead_axis(ax):
    ax.set_xlabel('Lead time (days)')
    ax.set_xticks(LEADS)
    ax.set_xticklabels(LEAD_LABELS)
    ax.grid(axis='y', linewidth=0.3, alpha=0.5)


def plot_acc_by_regime(acc, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), sharey=True)
    titles = ('(A) Water-limited (P/ET$_0$ < 0.5)', '(B) Energy-limited (P/ET$_0$ $\\geq$ 0.5)')
    for ax, regime, title in zip(axes, REGIMES, titles):
        for config in ORDER:
            ax.plot(LEADS, acc[config][regime], label=config, color=COLORS[config],
                    marker=MARKERS[config], markersize=5, linewidth=1.5,
                    linestyle=LINESTYLES[config])
        lead_axis(ax)
        ax.set_title(title, fontsize=10, loc='left')
        ax.axhline(0, color='grey', linewidth=0.5, zorder=0)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    values = np.concatenate([v for c in acc.values() for v in c.values()])
    axes[0].set_ylim(values.min() - 0.03, values.max() + 0.03)
    axes[0].set_ylabel('ACC')
    axes[1].legend(loc='upper right', framealpha=0.9)
    fig.tight_layout()
    save(fig, outdir, 'acc_by_aridity')


def plot_contribution_by_regime(acc_by_tlen, outdir):
    colors = {'water_limited': '#E66101', 'energy_limited': '#5E3C99'}
    names = {'water_limited': 'Water-limited', 'energy_limited': 'Energy-limited'}
    styles = {6: ('-', 'o'), 10: ('--', 'D')}
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for regime in REGIMES:
        for tlen, acc in acc_by_tlen.items():
            ls, marker = styles[tlen]
            ax.plot(LEADS, acc['NIRv+RZSM'][regime] - acc['NIRv-only'][regime],
                    color=colors[regime], linestyle=ls, marker=marker, markersize=5,
                    linewidth=1.5, label=f'{names[regime]}, $T_{{len}}$={tlen}')
    ax.axhline(0, color='grey', linewidth=0.8, zorder=0)
    lead_axis(ax)
    ax.set_ylabel('$\\Delta$ACC (NIRv+RZSM $-$ NIRv-only)')
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, outdir, 'rzsm_contribution_by_aridity')


def regime_table(acc) -> pd.DataFrame:
    rows = []
    for regime, short in REGIMES.items():
        for config in ORDER:
            values = acc[config][regime]
            rows.append([short, config, values.mean(), *values])
    for regime, short in REGIMES.items():
        delta = acc['NIRv+RZSM'][regime] - acc['NIRv-only'][regime]
        rows.append([f'delta {short}', 'RZSM - only', delta.mean(), *delta])
    columns = ['Regime', 'Configuration', 'Mean'] + [f't+{lead}' for lead in LEADS]
    return pd.DataFrame(rows, columns=columns).round(3)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='runs')
    p.add_argument('--outdir', default='figures')
    args = p.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    acc = {tlen: regime_acc(args.runs, tlen) for tlen in (6, 10)}
    table = regime_table(acc[6])
    print(table.to_string(index=False))
    table.to_csv(Path(args.outdir) / 'acc_by_aridity.csv', index=False)
    plot_acc_by_regime(acc[6], args.outdir)
    plot_contribution_by_regime(acc, args.outdir)


if __name__ == '__main__':
    main()
