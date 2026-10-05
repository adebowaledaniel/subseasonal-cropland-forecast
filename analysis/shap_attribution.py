"""
SHAP attribution of the NIRv+RZSM ConvLSTM.

GradientExplainer is applied to the spatially averaged forecast at each lead, with 100
background windows from the training years and every growing-season test window. Each
test window takes the dominant aridity class of its patch. Writes one summary per lead and
the figure; --plot_only redraws the figure from existing summaries.

    python analysis/shap_attribution.py --run runs/mse/nirv_rzsm_t10 --data_root work/zscore \
        --aridity_dir work/aridity --outdir figures/shap_t10
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import rasterio
import shap
import torch
import torch.nn as nn

from common import ROOT, load_model, save
from src.training import build_dataset, set_seed

CHANNELS = ['NIRv', 'RZSM']
REGIMES = {1: 'water_limited', 2: 'energy_limited'}


class LeadMean(nn.Module):
    """Spatial mean of the forecast at one lead, shape (B, 1)."""

    def __init__(self, model, lead: int):
        super().__init__()
        self.model, self.lead = model, lead

    def forward(self, x):
        return self.model(x)[:, self.lead, 0].mean(dim=(1, 2)).unsqueeze(1)


def dominant_class(aridity_dir, location):
    path = Path(aridity_dir) / location / 'binary_ai.tif'
    if not path.exists():
        return 0
    with rasterio.open(path) as src:
        classes = src.read(1)
    classes = classes[(classes == 1) | (classes == 2)].astype(int)
    return int(np.bincount(classes).argmax()) if classes.size else 0


def sample(dataset, n, seed, device):
    idx = np.random.RandomState(seed).choice(len(dataset), size=min(n, len(dataset)), replace=False)
    items = [dataset[i] for i in idx]
    return torch.stack([s['inputs'] for s in items]).to(device), [s['location'] for s in items]


def explain(model, background, inputs, batch_size=8):
    explainer = shap.GradientExplainer(model, background)
    values = []
    for start in range(0, len(inputs), batch_size):
        sv = np.asarray(explainer.shap_values(inputs[start:start + batch_size], nsamples=200))
        values.append(sv[..., 0] if sv.ndim == inputs.ndim + 1 else sv)
    return np.concatenate(values)


def summarise(values, regimes):
    """Mean |SHAP| per channel, over all windows and within each aridity regime."""
    magnitude = np.abs(values)
    out = {'n_samples': len(values), 'channel_importance': magnitude.mean(axis=(0, 1, 3, 4)).tolist(),
           'temporal_profile': magnitude.mean(axis=(0, 3, 4)).T.tolist()}
    for value, name in REGIMES.items():
        idx = regimes == value
        if idx.any():
            out[name] = {'n_samples': int(idx.sum()),
                         'channel_importance': magnitude[idx].mean(axis=(0, 1, 3, 4)).tolist()}
    return out


def rzsm_share(importance):
    return 100 * importance[1] / sum(importance)


def plot_attribution(summaries, outdir):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
                         'font.size': 8, 'axes.labelsize': 9, 'xtick.labelsize': 7.5,
                         'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.linewidth': 0.5,
                         'axes.spines.top': False, 'axes.spines.right': False})
    x = np.arange(len(summaries))
    labels = [f't+{i + 1}\n({(i + 1) * 8} d)' for i in x]
    rzsm = np.array([rzsm_share(s['channel_importance']) for s in summaries])
    wl = np.array([rzsm_share(s['water_limited']['channel_importance']) for s in summaries])
    el = np.array([rzsm_share(s['energy_limited']['channel_importance']) for s in summaries])

    fig = plt.figure(figsize=(8.5, 6.5))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.9], hspace=0.45)
    ax_a, ax_b = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])

    ax_a.bar(x, 100 - rzsm, 0.55, label='NIRv', color='#1a7a5a', edgecolor='white', linewidth=0.3)
    ax_a.bar(x, rzsm, 0.55, bottom=100 - rzsm, label='RZSM', color='#2e6fba', edgecolor='white', linewidth=0.3)
    for i, share in enumerate(rzsm):
        ax_a.text(i, 100 - share / 2, f'{share:.0f}%', ha='center', va='center', fontsize=7, color='white')
    ax_a.set_ylabel('Attribution share (%)')
    ax_a.set_ylim(0, 105)
    ax_a.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax_a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)
    ax_a.set_title('(a)', loc='left', fontsize=9)

    n_wl, n_el = summaries[0]['water_limited']['n_samples'], summaries[0]['energy_limited']['n_samples']
    ax_b.bar(x - 0.15, wl, 0.30, label=f'Water-limited (n={n_wl})', color='#b06a10', edgecolor='white', linewidth=0.3)
    ax_b.bar(x + 0.15, el, 0.30, label=f'Energy-limited (n={n_el})', color='#2e6fba', edgecolor='white', linewidth=0.3)
    for i in x:
        ax_b.annotate(f'{wl[i] - el[i]:+.1f}', xy=(i, max(wl[i], el[i]) + 0.7), ha='center', va='bottom',
                      fontsize=6.5, color='#666666')
    ax_b.set_ylabel('RZSM attribution share (%)')
    ax_b.set_ylim(12, 46)
    ax_b.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax_b.legend(loc='upper left', frameon=False)
    ax_b.set_title('(b)', loc='left', fontsize=9)
    ax_b.axhline(rzsm.mean(), ls=':', lw=0.6, color='#666666', zorder=0)
    ax_b.text(-0.45, rzsm.mean() + 0.4, f'{rzsm.mean():.1f}%', fontsize=6, color='#666666', va='bottom')

    for ax in (ax_a, ax_b):
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    save(fig, outdir, 'shap_attribution')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True, help='train.py output of the NIRv+RZSM model')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data_root', required=True)
    p.add_argument('--aridity_dir', required=True)
    p.add_argument('--growing_season', default=str(ROOT / 'configs/growing_season.yml'))
    p.add_argument('--n_background', type=int, default=100)
    p.add_argument('--outdir', default='figures/shap')
    p.add_argument('--plot_only', action='store_true')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        model, config = load_model(args.run, args.seed, args.device)
        root = Path(args.data_root)
        background, _ = sample(build_dataset(config, root / 'train', args.growing_season),
                               args.n_background, 42, args.device)
        test = build_dataset(config, root / 'test', args.growing_season)
        inputs, locations = sample(test, len(test), 123, args.device)
        regimes = np.array([dominant_class(args.aridity_dir, loc) for loc in locations])
        for lead in range(config['forecast_horizon']):
            set_seed(args.seed)
            values = explain(LeadMean(model, lead), background, inputs)
            summary = summarise(values, regimes)
            with open(outdir / f'lead_{lead + 1}.json', 'w') as f:
                json.dump(summary, f, indent=1)
            print(f"t+{lead + 1}: RZSM share {rzsm_share(summary['channel_importance']):.1f}%")

    summaries = [json.load(open(f)) for f in sorted(outdir.glob('lead_*.json'))]
    plot_attribution(summaries, outdir)


if __name__ == '__main__':
    main()
