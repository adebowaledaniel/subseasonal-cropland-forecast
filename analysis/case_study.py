"""
Case study of the 2024 southern African El Nino drought.

Every southern African patch is forecast for four verification composites (DOY 81-105,
2024) from the window ending `lead` composites earlier; inputs that start in 2023 come from
the validation split. Forecasts are mosaicked on the 9 km grid and scored inside two
boxes with MAE, Pearson r and SSIM.

    python analysis/case_study.py --runs runs --data_root work/zscore \
        --cropland_masks work/cropland_masks --outdir figures/case_study
"""

import argparse
import csv
from datetime import date, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from pyproj import Transformer
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity

from common import ROOT, load_model, mosaic, save
from src.training import build_dataset

CASE_RUNS = {'NIRv+RZSM': 'mse/nirv_rzsm_t10', 'NIRv-only': 'mse/nirv_only_t10',
             'NIRv+lagRZSM': 'mse/nirv_lagrzsm_t10'}
COUNTRIES = {'South Africa', 'Zambia', 'Zimbabwe', 'Mozambique', 'Botswana'}
BOXES = {'ZWE/ZMB': (25.528278, -22.398332, 33.042927, -15.567482),
         'ZAF/LSO': (23.827972, -30.652828, 30.496674, -25.835278)}
BOX_COLORS = {'ZWE/ZMB': '#e63946', 'ZAF/LSO': '#457b9d'}
VERIFICATION_DOYS = (81, 89, 97, 105)
YEAR = 2024
EASE2 = 'EPSG:6933'


def case_dataset(config, data_root, locations):
    """
    Every window of the validation and test years, without season or validity filters.
    As published, pixels without a significant lag receive zero in the lagged channel.
    """
    return build_dataset(dict(config, growing_season_only=False, min_valid_ratio=0.0),
                         [data_root / 'val', data_root / 'test'], locations=locations,
                         missing_lag='zero')


def forecast_maps(model, config, dataset, day, lead):
    """Per-location forecast and observation (H, W) for one verification composite."""
    period = (day - 1) // 8
    picks = {}
    for i, (loc, start) in enumerate(dataset.index):
        s = dataset.series[loc]
        t = start + config['tlen'] + lead
        if loc not in picks and t < len(s['doy']) and s['doy'][t] == period and s['years'][t] == YEAR:
            picks[loc] = i
    batch = [dataset[i] for i in picks.values()]
    with torch.no_grad():
        pred = model(torch.stack([b['inputs'] for b in batch]))[:, lead, 0].numpy()
    obs = [np.where(b['target_mask'][lead, 0].numpy(), b['target'][lead, 0].numpy(), np.nan) for b in batch]
    return dict(zip(picks, pred)), dict(zip(picks, obs))


def box_pixels(box, transform, shape):
    to_ease = Transformer.from_crs('EPSG:4326', EASE2, always_xy=True)
    lon0, lat0, lon1, lat1 = box
    (x0, x1), (y0, y1) = zip(to_ease.transform(lon0, lat0), to_ease.transform(lon1, lat1))
    c0, r0 = ~transform * (min(x0, x1), max(y0, y1))
    c1, r1 = ~transform * (max(x0, x1), min(y0, y1))
    return max(0, int(r0)), min(shape[0], int(r1)), max(0, int(c0)), min(shape[1], int(c1))


def box_metrics(pred, obs, rows_cols):
    r0, r1, c0, c1 = rows_cols
    p, o = pred[r0:r1, c0:c1], obs[r0:r1, c0:c1]
    valid = ~np.isnan(p) & ~np.isnan(o)
    if valid.sum() < 4:
        return {'MAE': np.nan, 'SSIM': np.nan, 'r': np.nan, 'n': int(valid.sum())}
    ssim = structural_similarity(np.nan_to_num(p), np.nan_to_num(o), data_range=6.0, win_size=7)
    return {'MAE': float(np.mean(np.abs(p[valid] - o[valid]))), 'SSIM': float(ssim),
            'r': float(pearsonr(p[valid], o[valid])[0]), 'n': int(valid.sum())}


def figure(rows, transform, lead, outdir):
    """rows: list of (init date, verification date, {column: mosaic}) for one lead."""
    shape = next(iter(rows[0][2].values())).shape
    left, bottom, right, top = rasterio.transform.array_bounds(*shape, transform)
    to_lonlat = Transformer.from_crs(EASE2, 'EPSG:4326', always_xy=True)
    (lon0, lon1), (lat0, lat1) = zip(to_lonlat.transform(left, bottom), to_lonlat.transform(right, top))
    pad_lon, pad_lat = (lon1 - lon0) * 0.03, (lat1 - lat0) * 0.03
    extent = [lon0 - pad_lon, lon1 + pad_lon, lat0 - pad_lat, lat1 + pad_lat]
    boxes = {k: box_pixels(v, transform, shape) for k, v in BOXES.items()}

    columns = list(rows[0][2])
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(2.5 * len(columns) + 1, 3.0 * len(rows) + 0.5),
                             subplot_kw={'projection': ccrs.PlateCarree()},
                             gridspec_kw={'wspace': 0.1, 'hspace': 0.0}, squeeze=False)
    cmap, norm = plt.get_cmap('BrBG'), mpl.colors.Normalize(vmin=-3, vmax=3)
    for r, (init, verif, maps) in enumerate(rows):
        for c, (label, values) in enumerate(maps.items()):
            ax = axes[r, c]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.set_aspect('auto')
            ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none', zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor='#e8f0fe', edgecolor='none', zorder=0)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='#777777', zorder=3)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#555555', zorder=3)
            ax.imshow(np.ma.masked_invalid(values), origin='upper', extent=[left, right, bottom, top],
                      transform=ccrs.epsg(6933), cmap=cmap, norm=norm, interpolation='nearest', zorder=2)
            for name, (b_lon0, b_lat0, b_lon1, b_lat1) in BOXES.items():
                ax.add_patch(mpatches.Rectangle((b_lon0, b_lat0), b_lon1 - b_lon0, b_lat1 - b_lat0,
                                                linewidth=1.8, edgecolor=BOX_COLORS[name], facecolor='none',
                                                transform=ccrs.PlateCarree(), zorder=5))
                ax.text(b_lon0, b_lat1 + 0.3, name, fontsize=6, fontweight='bold', color=BOX_COLORS[name],
                        transform=ccrs.PlateCarree(), zorder=6,
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))
            if c > 0:
                text = '\n'.join(f"{name}: MAE={m['MAE']:.2f} SSIM={m['SSIM']:.2f} r={m['r']:.2f}"
                                 for name, m in ((n, box_metrics(values, maps['Observed'], b))
                                                 for n, b in boxes.items()))
                ax.text(0.98, 0.02, text, fontsize=5, family='monospace', transform=ax.transAxes,
                        va='bottom', ha='right', zorder=7,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.85, lw=0.5))
            if r == 0:
                ax.set_title(label, fontsize=10, fontweight='bold', pad=6)
            gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.3, linestyle='--', zorder=4)
            gl.top_labels = gl.right_labels = False
            gl.left_labels, gl.bottom_labels = c == 0, r == len(rows) - 1
            gl.xlabel_style = gl.ylabel_style = {'size': 6}
        axes[r, 0].text(-0.15, 0.5, f'Init: {init}\nVerif: {verif}\nLead: t+{lead} ({lead * 8}d)',
                        fontsize=7, fontweight='bold', transform=axes[r, 0].transAxes,
                        va='center', ha='right', linespacing=1.6)

    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([0.15, 0.02, 0.7, 0.015]),
                       orientation='horizontal')
    bar.set_label('NIRv anomaly (z-score)', fontsize=9, labelpad=4)
    fig.legend(handles=[mpatches.Patch(edgecolor=BOX_COLORS[n], facecolor='none', linewidth=2, label=n)
                        for n in BOXES], loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=2,
               fontsize=7, frameon=True, edgecolor='gray')
    plt.subplots_adjust(left=0.1, right=0.99, bottom=0.07, wspace=0.1, hspace=0.0)
    save(fig, outdir, f'case_study_lead{lead}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='runs')
    p.add_argument('--data_root', required=True)
    p.add_argument('--cropland_masks', required=True)
    p.add_argument('--grid_attributes', default=str(ROOT / 'data/grid_attributes.csv'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--outdir', default='figures/case_study')
    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    with open(args.grid_attributes) as f:
        countries = {row['Name']: row['country'] for row in csv.DictReader(f)}
    locations = sorted(d.name for d in (data_root / 'test').iterdir()
                       if d.is_dir() and countries.get(d.name) in COUNTRIES)
    models = {label: load_model(Path(args.runs) / run, args.seed) for label, run in CASE_RUNS.items()}
    datasets = {label: case_dataset(config, data_root, locations) for label, (_, config) in models.items()}

    records = []
    for lead in range(1, 6):
        rows = []
        for day in VERIFICATION_DOYS:
            verif = date(YEAR, 1, 1) + timedelta(days=day - 1)
            maps, transform = {}, None
            for label, (model, config) in models.items():
                pred, obs = forecast_maps(model, config, datasets[label], day, lead - 1)
                if 'Observed' not in maps:
                    maps['Observed'], transform = mosaic(obs, args.cropland_masks)
                maps[label], _ = mosaic(pred, args.cropland_masks)
            rows.append(((verif - timedelta(days=8 * lead)).isoformat(), verif.isoformat(), maps))
            for label in CASE_RUNS:
                for name, box in BOXES.items():
                    m = box_metrics(maps[label], maps['Observed'], box_pixels(box, transform, maps[label].shape))
                    records.append({'lead': lead, 'verification': verif.isoformat(), 'model': label,
                                    'box': name, **m})
        figure(rows, transform, lead, outdir)

    table = pd.DataFrame(records)
    table.to_csv(outdir / 'case_study_metrics.csv', index=False)
    summary = table.groupby(['lead', 'model', 'box'], sort=False)[['r', 'MAE', 'SSIM']].mean().round(3)
    summary.to_csv(outdir / 'case_study_summary.csv')
    print(summary.unstack('box').to_string())


if __name__ == '__main__':
    main()
