"""
Per-pixel ACC over all growing-season forecasts of the test years, pooled over
leads, for each configuration (A-C) and the NIRv+RZSM improvement maps (D-E).

Forecast windows here are every window whose target composites fall in the growing
season, with no minimum-valid-data rule, so all cropland patches contribute.

    python analysis/skill_maps.py --runs runs --data_root work/zscore \
        --cropland_masks work/cropland_masks --outdir figures
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from common import ROOT, load_model, mosaic, save
from src.evaluation import predict
from src.metrics import pixelwise_acc
from src.training import build_dataset, make_loader

MAP_RUNS = {'NIRv+RZSM': 'mse/nirv_rzsm_t6', 'NIRv-only': 'mse/nirv_only_t6',
            'NIRv+lagRZSM': 'mse/nirv_lagrzsm_t6'}


def location_acc_maps(model, config, split_dir, growing_season, device='cpu', min_samples=5):
    """{location: (H, W) per-pixel ACC} pooled over forecasts and leads."""
    # As published, pixels without a significant lag received zero in the lagged channel.
    dataset = build_dataset(config, split_dir, growing_season, season_filter='target',
                            min_valid_ratio=0.0, missing_lag='zero')
    out = predict(model, make_loader(dataset, dict(config, num_workers=0)), device)
    obs = np.where(out['mask'], out['obs'], np.nan)
    locations = np.array(out['location'])
    maps = {}
    for loc in dict.fromkeys(out['location']):
        idx = locations == loc
        h, w = obs.shape[-2:]
        pred, ob = out['pred'][idx].reshape(-1, h, w), obs[idx].reshape(-1, h, w)
        if len(pred) >= min_samples:
            maps[loc] = pixelwise_acc(pred, ob, min_samples)
    return maps


def lonlat_extent(bounds, pad=0.03):
    left, bottom, right, top = bounds
    to_lonlat = Transformer.from_crs('EPSG:6933', 'EPSG:4326', always_xy=True)
    (lon0, lon1), (lat0, lat1) = zip(*(to_lonlat.transform(left, bottom), to_lonlat.transform(right, top)))
    dx, dy = (lon1 - lon0) * pad, (lat1 - lat0) * pad
    return [lon0 - dx, lon1 + dx, lat0 - dy, lat1 + dy]


def map_panel(ax, values, extent, cmap, norm, title, stats):
    image_extent = [extent[0] + (extent[1] - extent[0]) * 0.03, extent[1] - (extent[1] - extent[0]) * 0.03,
                    extent[2] + (extent[3] - extent[2]) * 0.03, extent[3] - (extent[3] - extent[2]) * 0.03]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#e8f0fe', edgecolor='none', zorder=0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='#777777', zorder=3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#555555', zorder=3)
    ax.imshow(np.ma.masked_invalid(values), origin='upper', extent=image_extent,
              transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, interpolation='nearest', zorder=2)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
    ax.text(0.98, 0.02, stats, fontsize=6, family='monospace', transform=ax.transAxes,
            va='bottom', ha='right', zorder=7,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.85, lw=0.5))
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.3, linestyle='--', zorder=4)
    gl.top_labels = gl.right_labels = False
    gl.xformatter = gl.yformatter = mticker.FormatStrFormatter('%.0f°')
    gl.xlabel_style = gl.ylabel_style = {'size': 6}


def plot_skill_maps(acc, bounds, outdir):
    labels = list(acc)
    extent = lonlat_extent(bounds)
    fig = plt.figure(figsize=(2.5 * len(labels) + 1.8, 9.5))
    top = fig.add_gridspec(2, len(labels))
    bottom = fig.add_gridspec(2, 2 * (len(labels) - 1), left=0.25, right=0.80, wspace=0.05)

    cmap, norm = plt.get_cmap('RdYlBu_r'), mpl.colors.Normalize(vmin=-0.4, vmax=0.9)
    for i, label in enumerate(labels):
        v = acc[label][~np.isnan(acc[label])]
        stats = f'med={np.median(v):.2f}\n>0.6: {np.mean(v > 0.6) * 100:.0f}%\n>0.5: {np.mean(v > 0.5) * 100:.0f}%'
        ax = fig.add_subplot(top[0, i], projection=ccrs.PlateCarree())
        map_panel(ax, acc[label], extent, cmap, norm, f'[{chr(65 + i)}] {label}', stats)
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([0.92, 0.53, 0.015, 0.35]))
    bar.set_label('ACC', fontsize=9)

    cmap, norm = plt.get_cmap('RdBu'), mpl.colors.TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=0.3)
    for j, other in enumerate(labels[1:]):
        diff = acc[labels[0]] - acc[other]
        d = diff[~np.isnan(diff)]
        stats = f'med={np.median(d):+.3f}\n+ve: {np.mean(d > 0) * 100:.0f}%'
        ax = fig.add_subplot(bottom[1, 2 * j:2 * j + 2], projection=ccrs.PlateCarree())
        map_panel(ax, diff, extent, cmap, norm, f'[{chr(65 + len(labels) + j)}] {labels[0]} - {other}', stats)
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([0.92, 0.08, 0.015, 0.35]))
    bar.set_label('ACC Improvement', fontsize=9)
    save(fig, outdir, 'skill_maps')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='runs')
    p.add_argument('--data_root', required=True)
    p.add_argument('--cropland_masks', required=True, help='output of preprocessing/cropland_mask.py')
    p.add_argument('--growing_season', default=str(ROOT / 'configs/growing_season.yml'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--outdir', default='figures')
    args = p.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    acc, rows = {}, []
    for label, run in MAP_RUNS.items():
        model, config = load_model(Path(args.runs) / run, args.seed)
        maps = location_acc_maps(model, config, Path(args.data_root) / 'test', args.growing_season)
        acc[label], transform = mosaic(maps, args.cropland_masks)
        v = acc[label][~np.isnan(acc[label])]
        rows.append({'model': label, 'pixels': v.size, 'mean': v.mean(), 'median': np.median(v),
                     '>0.5 (%)': np.mean(v > 0.5) * 100, '>0.6 (%)': np.mean(v > 0.6) * 100})
        np.save(Path(args.outdir) / f"acc_map_{label.replace('+', '_').replace('-', '_')}.npy", acc[label])

    table = pd.DataFrame(rows).round(4)
    print(table.to_string(index=False))
    table.to_csv(Path(args.outdir) / 'skill_map_summary.csv', index=False)
    bounds = rasterio.transform.array_bounds(*acc['NIRv+RZSM'].shape, transform)
    plot_skill_maps(acc, bounds, args.outdir)


if __name__ == '__main__':
    main()
