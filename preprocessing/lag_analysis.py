"""
Pixel-wise RZSM-vegetation coupling and optimal lag.

For each cell, the training-year growing-season composites of the vegetation index and of
RZSM are rank-normalised by day of year. RZSM is shifted by 1..max_lag composites and the
Spearman correlation at each lag is tested with the autocorrelation-corrected test, at
alpha / max_lag (Bonferroni). The optimal lag is the significant lag with the largest
|rho|.

Writes {output_dir}/{LOC}_{VAR}_sm_rootzone.tif with bands optimal_lag, correlation and
p_value (-9999 where no lag is significant). With --data_root the lag map is also saved in
every split as {LOC}_{VAR}_sm_rootzone.npy, which the lagged-RZSM input reads.

    python preprocessing/lag_analysis.py --composites work/composites --output_dir work/lags \
        --data_root work/zscore --target NIRV

The published coupling maps and the coupling covariate of the skill-gain regression come
from an earlier run that kept lags significant at p < 0.05 without the Bonferroni
correction, on an earlier version of the composites:

    python preprocessing/lag_analysis.py --composites work/composites --output_dir work/lags_p05 \
        --alpha 0.05 --no-bonferroni
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
import yaml

from rank_correlation import rank_by_day_of_year, spearman_srd_test
from standardize import load_composites

BANDS = {'NDVI': 0, 'NIRV': 1}
RZSM = 3
NODATA = -9999.0


def lag_maps(dates, data, target_band, months, years, max_lag, alpha, min_obs, bonferroni=True):
    keep = np.array([d.year in years and (months is None or d.month in months) for d in dates])
    doy = np.array([d.timetuple().tm_yday for d, k in zip(dates, keep) if k])
    target = rank_by_day_of_year(data[keep, target_band], doy)
    rzsm = rank_by_day_of_year(data[keep, RZSM], doy)

    threshold = alpha / max_lag if bonferroni else alpha
    best = np.full(target.shape[1:], NODATA)
    lag_map, p_map = best.copy(), best.copy()
    for lag in range(1, max_lag + 1):
        shifted = np.full_like(rzsm, np.nan)
        shifted[lag:] = rzsm[:-lag]
        for i, j in np.ndindex(*best.shape):
            valid = ~np.isnan(target[:, i, j]) & ~np.isnan(shifted[:, i, j])
            if valid.sum() < min_obs:
                continue
            rho, p = spearman_srd_test(shifted[valid, i, j], target[valid, i, j])
            if np.isfinite(rho) and abs(rho) > best[i, j] and p < threshold:
                best[i, j], lag_map[i, j], p_map[i, j] = abs(rho), lag, p
    return lag_map, best, p_map


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--composites', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--data_root', help='standardised data to receive the lag maps')
    p.add_argument('--target', choices=BANDS, default='NIRV')
    p.add_argument('--growing_season', default='configs/growing_season.yml')
    p.add_argument('--train_years', type=int, nargs=2, default=[2016, 2022])
    p.add_argument('--max_lag', type=int, default=5)
    p.add_argument('--alpha', type=float, default=0.01)
    p.add_argument('--bonferroni', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--min_obs', type=int, default=100)
    args = p.parse_args()

    with open(args.growing_season) as f:
        seasons = yaml.safe_load(f)
    years = range(args.train_years[0], args.train_years[1] + 1)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for loc_dir in sorted(d for d in Path(args.composites).iterdir() if d.is_dir()):
        loc = loc_dir.name
        dates, data = load_composites(loc_dir)
        maps = lag_maps(dates, data, BANDS[args.target], seasons.get(loc), years,
                        args.max_lag, args.alpha, args.min_obs, args.bonferroni)
        name = f'{loc}_{args.target}_sm_rootzone'

        with rasterio.open(next(loc_dir.glob('*.tif'))) as src:
            profile = dict(driver='GTiff', height=src.height, width=src.width, count=3,
                           dtype='float32', crs=src.crs, transform=src.transform, nodata=NODATA)
        with rasterio.open(out / f'{name}.tif', 'w', **profile) as dst:
            dst.write(np.stack(maps).astype(np.float32))
            for band, label in enumerate(('optimal_lag', 'correlation', 'p_value'), start=1):
                dst.set_band_description(band, label)

        if args.data_root:
            for split_dir in Path(args.data_root).iterdir():
                if (split_dir / loc).is_dir():
                    np.save(split_dir / loc / f'{name}.npy', maps[0].astype(np.float32))
        print(f'{loc}: {np.mean(maps[0] != NODATA):.0%} of pixels with a significant lag')


if __name__ == '__main__':
    main()
