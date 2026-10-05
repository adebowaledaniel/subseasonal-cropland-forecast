"""
Daily patch stacks -> 8-day median composites masked to consistent cropland.

Composites start on 1 January of each year (46 per year, the last one shorter) and take
the per-pixel median of the available days. Cells whose consistent cropland covers less
than `min_cropland` of the patch are dropped. Writes {output_dir}/{LOC}/{YYYY-MM-DD}.tif
with bands NDVI, NIRv, surface SM and root-zone SM.

    python preprocessing/composite.py --patches work/patches \
        --cropland_masks work/cropland_masks --output_dir work/composites
"""

import argparse
import re
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio

VARIABLES = ['NDVI', 'NIRV', 'sm_surface', 'sm_rootzone']


def daily_bands(descriptions):
    """{date: {variable: band index}} from band names such as NIRV_2017-02-01."""
    bands = defaultdict(dict)
    for i, name in enumerate(descriptions):
        m = re.fullmatch(r'(.+)_(\d{4}-\d{2}-\d{2})', name or '')
        if m:
            bands[date.fromisoformat(m.group(2))][m.group(1)] = i
    return bands


def composite_periods(year: int, interval: int):
    start, end = date(year, 1, 1), date(year, 12, 31)
    while start <= end:
        yield start, min(start + timedelta(days=interval - 1), end)
        start += timedelta(days=interval)


def composite_year(path: Path, interval: int, mask: np.ndarray, year: int):
    with rasterio.open(path) as src:
        data = src.read()
        bands = daily_bands(src.descriptions)
        profile = dict(driver='GTiff', height=src.height, width=src.width, count=len(VARIABLES),
                       dtype='float32', crs=src.crs, transform=src.transform)
    for start, end in composite_periods(year, interval):
        days = [d for d in sorted(bands) if start <= d <= end]
        if not days:
            continue
        layers = []
        for var in VARIABLES:
            stack = [data[bands[d][var]] for d in days if var in bands[d]]
            layers.append(np.nanmedian(stack, axis=0) if stack
                          else np.full(data.shape[1:], np.nan, dtype=np.float32))
        composite = np.stack(layers).astype(np.float32)
        composite[:, mask != 1] = np.nan
        yield start, composite, profile


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--patches', required=True, help='directory of {LOC}_viirs_{YEAR}.tif')
    p.add_argument('--cropland_masks', required=True, help='output of cropland_mask.py')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--interval', type=int, default=8)
    p.add_argument('--min_cropland', type=float, default=0.10)
    args = p.parse_args()

    files = defaultdict(list)
    for f in sorted(Path(args.patches).glob('*.tif')):
        m = re.fullmatch(r'(\w+_G_\d+)_viirs_(\d{4})', f.stem)
        if m:
            files[m.group(1)].append((int(m.group(2)), f))

    kept = 0
    for loc, year_files in sorted(files.items()):
        with rasterio.open(Path(args.cropland_masks) / loc / 'consistent_cropland.tif') as src:
            mask = src.read(1)
        if np.mean(mask == 1) < args.min_cropland:
            continue
        out_dir = Path(args.output_dir) / loc
        out_dir.mkdir(parents=True, exist_ok=True)
        for year, path in sorted(year_files):
            for start, composite, profile in composite_year(path, args.interval, mask, year):
                with rasterio.open(out_dir / f'{start.isoformat()}.tif', 'w', **profile) as dst:
                    dst.write(composite)
        kept += 1
    print(f'{kept} of {len(files)} cells kept (consistent cropland >= {args.min_cropland:.0%})')


if __name__ == '__main__':
    main()
