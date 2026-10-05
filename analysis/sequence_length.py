"""
Forecast skill as a function of the input window T in {2, ..., 12}.
Domain ACC is pooled over leads; regime ACC is the mean of the per-lead values.

    python analysis/sequence_length.py --runs runs --outdir figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from common import CONFIGS, DOMAIN_RUNS, REGIME_RUNS, TLENS, load_runs, save, seed_stats

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 10, 'legend.fontsize': 8,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.linewidth': 0.6,
    'lines.linewidth': 1.3, 'lines.markersize': 5,
})

STYLE = {'NIRv+RZSM': ('#1f77b4', 'o', '-'), 'NIRv-only': ('#d62728', 's', '--'),
         'NIRv+lagRZSM': ('#2ca02c', '^', ':')}
REGIME_STYLE = {'water_limited': ('#d4880f', 'o', '-', 'Water-limited'),
                'energy_limited': ('#4a8fa8', 's', '--', 'Energy-limited')}
DAYS = np.array(TLENS) * 8


def collect(runs_root):
    """Seed mean and SD per (config, T) of domain ACC and of the lead-mean regime ACC."""
    out = {}
    for config in CONFIGS:
        for tlen in TLENS:
            domain = load_runs(runs_root, DOMAIN_RUNS[config], tlen)
            regime = load_runs(runs_root, REGIME_RUNS[config], tlen)
            entry = {'acc': seed_stats(domain, 'acc')}
            for r in REGIME_STYLE:
                lead_means = np.array([np.mean(run[f'{r}_acc_lead']) for run in regime])
                entry[r] = (lead_means.mean(), lead_means.std())
            out[config, tlen] = entry
    return out


def smooth(x, y, num=300, window=35):
    """Hann-smoothed dense curve, used only for the shading."""
    xd = np.linspace(x.min(), x.max(), num)
    yd = np.interp(xd, x, y)
    kernel = np.hanning(window) / np.hanning(window).sum()
    return xd, np.convolve(np.pad(yd, window // 2, mode='edge'), kernel, mode='valid')


def band(ax, mean, sd, color):
    xd, lower = smooth(DAYS, mean - sd)
    _, upper = smooth(DAYS, mean + sd)
    ax.fill_between(xd, lower, upper, color=color, alpha=0.12, zorder=1)


def plot_sequence_length(stats, outdir):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.5, 3.0), gridspec_kw={'wspace': 0.38})

    for config, (color, marker, ls) in STYLE.items():
        mean = np.array([stats[config, t]['acc'][0] for t in TLENS])
        sd = np.array([stats[config, t]['acc'][1] for t in TLENS])
        ax_a.plot(DAYS, mean, color=color, marker=marker, ls=ls, label=config, zorder=3)
        band(ax_a, mean, sd, color)
    ax_a.axvline(48, color='grey', ls='--', lw=0.7, alpha=0.5, zorder=0)
    ax_a.annotate(r'$T_{\mathrm{len}}{=}6$', xy=(49, 0.567), fontsize=7.5, color='grey', va='top')
    ax_a.set_ylabel('ACC (domain-averaged)')
    ax_a.set_ylim(0.500, 0.570)
    ax_a.yaxis.set_major_locator(MultipleLocator(0.01))

    for regime, (color, marker, ls, label) in REGIME_STYLE.items():
        rzsm = np.array([stats['NIRv+RZSM', t][regime] for t in TLENS])
        only = np.array([stats['NIRv-only', t][regime] for t in TLENS])
        delta, sd = rzsm[:, 0] - only[:, 0], np.hypot(rzsm[:, 1], only[:, 1])
        face = color if regime == 'water_limited' else 'white'
        ax_b.plot(DAYS, delta, color=color, marker=marker, ls=ls, markerfacecolor=face,
                  label=label, zorder=3)
        band(ax_b, delta, sd, color)
    ax_b.axhline(0, color='black', lw=0.6, zorder=2)
    ax_b.set_ylabel(r'$\Delta$ACC (NIRv+RZSM $-$ NIRv-only)')
    ax_b.set_ylim(-0.025, 0.055)

    for ax, letter in ((ax_a, 'A'), (ax_b, 'B')):
        ax.set_xlabel('Input context (days)')
        ax.set_xticks(DAYS)
        ax.legend(loc='lower left', framealpha=0.85, edgecolor='none')
        ax.grid(True, alpha=0.2, lw=0.4)
        ax.set_title(letter, loc='left', fontweight='bold', fontsize=11)
    save(fig, outdir, 'sequence_length')


def sequence_length_table(stats) -> pd.DataFrame:
    rows = []
    for tlen in TLENS:
        row = {'T': tlen}
        for key, prefix in (('acc', 'ACC'), ('water_limited', 'WL ACC'), ('energy_limited', 'EL ACC')):
            for config in ('NIRv+RZSM', 'NIRv-only', 'NIRv+lagRZSM'):
                row[f'{prefix} {config}'] = round(stats[config, tlen][key][0], 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='runs')
    p.add_argument('--outdir', default='figures')
    args = p.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    stats = collect(args.runs)
    table = sequence_length_table(stats)
    print(table.to_string(index=False))
    table.to_csv(Path(args.outdir) / 'sequence_length.csv', index=False)
    plot_sequence_length(stats, args.outdir)


if __name__ == '__main__':
    main()
