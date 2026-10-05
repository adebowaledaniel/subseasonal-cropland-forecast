"""
Cropland masks for each grid cell from the annual cropland maps of Khan et al. (2026).

The 30 m annual maps (0 = other, 100 = cropland) are resampled bilinearly onto each
cell's 32 x 32 grid (9 km, EPSG:6933), which gives a cropland percentage per pixel.
Pixels with at least 10 % cropland are cropland in that year, and the consistent mask
keeps pixels that are cropland in the majority of years. The percentages are also kept
(cropland_percent_{year}.tif); the per-pixel skill maps are masked with the last year.

    python preprocessing/cropland_mask.py --grids data/grid/study_grid.shp \
        --cropland cropland/africa_{2016..2024}.tif --output_dir work/cropland_masks
"""

import argparse
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds as window_from_bounds

EASE2 = 'EPSG:6933'
SIZE = 32


def resample_to_cell(src, bounds) -> np.ndarray:
    window = window_from_bounds(*transform_bounds(EASE2, src.crs, *bounds), transform=src.transform)
    data = src.read(1, window=window, boundless=True)
    out = np.zeros((SIZE, SIZE), dtype=src.dtypes[0])
    reproject(data, out, src_transform=src.window_transform(window), src_crs=src.crs,
              dst_transform=from_bounds(*bounds, SIZE, SIZE), dst_crs=EASE2,
              resampling=Resampling.bilinear)
    return out


def write(path: Path, array: np.ndarray, bounds) -> None:
    profile = dict(driver='GTiff', height=SIZE, width=SIZE, count=1, dtype=array.dtype,
                   crs=EASE2, transform=from_bounds(*bounds, SIZE, SIZE))
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--grids', required=True)
    p.add_argument('--cropland', nargs='+', required=True, help='annual maps, year in the file name')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--threshold', type=float, default=10, help='minimum cropland percentage')
    args = p.parse_args()

    years = {int(re.search(r'(\d{4})', Path(f).stem).group(1)): f for f in args.cropland}
    sources = {year: rasterio.open(f) for year, f in sorted(years.items())}
    grids = gpd.read_file(args.grids).to_crs(EASE2)

    for _, cell in grids.iterrows():
        out_dir = Path(args.output_dir) / cell.Name
        out_dir.mkdir(parents=True, exist_ok=True)
        bounds = cell.geometry.bounds
        annual = []
        for year, src in sources.items():
            percent = resample_to_cell(src, bounds)
            mask = (percent >= args.threshold).astype(np.uint8)
            write(out_dir / f'cropland_percent_{year}.tif', percent, bounds)
            write(out_dir / f'cropland_{year}.tif', mask, bounds)
            annual.append(mask)
        consistent = (np.mean(annual, axis=0) > 0.5).astype(np.uint8)
        write(out_dir / 'consistent_cropland.tif', consistent, bounds)
        print(f'{cell.Name}: {consistent.mean():.1%} consistent cropland')


if __name__ == '__main__':
    main()
