"""
RZSM-NIRv coupling. Mosaics the per-cell outputs of preprocessing/lag_analysis.py into one
EPSG:4326 GeoTIFF (bands optimal_lag, correlation, p_value) for mapping, and plots the
distribution of the maximum cross-correlation in water- and energy-limited cropland.

    python analysis/coupling_maps.py --lags work/lags_p05 --aridity work/aridity --outdir figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from matplotlib.patches import Rectangle
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject

from common import save

NODATA = -9999.0
COLORS = {'Water Limited': '#dec54a', 'Energy Limited': '#90dfe3'}


def mosaic_to_4326(lag_files, out_path):
    sources = [rasterio.open(f) for f in lag_files]
    array, transform = merge(sources, nodata=NODATA)
    crs = sources[0].crs
    height, width = array.shape[1:]
    dst_transform, dst_width, dst_height = calculate_default_transform(
        crs, 'EPSG:4326', width, height,
        *rasterio.transform.array_bounds(height, width, transform))
    out = np.full((array.shape[0], dst_height, dst_width), NODATA, dtype=np.float32)
    for band in range(array.shape[0]):
        reproject(array[band], out[band], src_transform=transform, src_crs=crs,
                  src_nodata=NODATA, dst_transform=dst_transform, dst_crs='EPSG:4326',
                  dst_nodata=NODATA, resampling=Resampling.nearest)
    profile = dict(driver='GTiff', height=dst_height, width=dst_width, count=array.shape[0],
                   dtype='float32', crs='EPSG:4326', transform=dst_transform, nodata=NODATA,
                   compress='lzw')
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(out)
        for band, name in enumerate(('optimal_lag', 'correlation', 'p_value'), start=1):
            dst.set_band_description(band, name)


def correlation_by_regime(lag_files, aridity_dir):
    values = {'Water Limited': [], 'Energy Limited': []}
    for f in lag_files:
        loc = f.name.split('_NIRV')[0].split('_NDVI')[0]
        aridity_file = Path(aridity_dir) / loc / 'binary_ai.tif'
        if not aridity_file.exists():
            continue
        with rasterio.open(f) as src:
            corr = src.read(2)
        with rasterio.open(aridity_file) as src:
            classes = src.read(1)
        significant = corr != NODATA
        values['Water Limited'].append(corr[significant & (classes == 1)])
        values['Energy Limited'].append(corr[significant & (classes == 2)])
    return {k: np.concatenate(v) for k, v in values.items()}


def violin(values, outdir):
    data = pd.concat([pd.DataFrame({'Cross-Correlation': v, 'category': k}) for k, v in values.items()])
    data['landcover'] = 'Cropland'
    sns.set_theme(style='white')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'axes.labelsize': 25, 'xtick.labelsize': 15, 'ytick.labelsize': 20,
                         'legend.fontsize': 17})
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(x='landcover', y='Cross-Correlation', data=data, inner=None, hue='category',
                   split=True, fill=True, bw_adjust=1.5, palette=COLORS, legend=False, ax=ax)
    ax.hlines(np.median(values['Water Limited']), -0.382, 0, colors='black', linewidth=1.7)
    ax.hlines(np.median(values['Energy Limited']), 0, 0.384, colors='black', linewidth=1.5)
    ax.set_xlabel('Cropland')
    ax.set_xticks([])
    for side in ('top', 'right', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=COLORS['Water Limited'], ec='black',
                                 label='Water-limited\n(P/PET < 0.5)'),
                       Rectangle((0, 0), 1, 1, fc=COLORS['Energy Limited'], ec='black',
                                 label='Energy-limited\n(P/PET > 0.5)')],
              loc='upper right', frameon=False, handlelength=1.5, handleheight=1.5)
    save(fig, outdir, 'coupling_by_aridity')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lags', required=True, help='output directory of lag_analysis.py')
    p.add_argument('--aridity', required=True, help='output directory of aridity.py')
    p.add_argument('--outdir', default='figures')
    args = p.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    lag_files = sorted(Path(args.lags).glob('*_sm_rootzone.tif'))
    mosaic_to_4326(lag_files, Path(args.outdir) / 'coupling_4326.tif')
    values = correlation_by_regime(lag_files, args.aridity)
    for regime, v in values.items():
        print(f'{regime}: {v.size} pixels, median correlation {np.median(v):.3f}')
    violin(values, args.outdir)


if __name__ == '__main__':
    main()
