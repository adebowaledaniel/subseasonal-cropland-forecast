"""
8-day composites -> standardised day-of-year anomalies split by year.

1. Climatology: mean of each composite period per pixel and band over the training years.
2. Anomalies: composite minus its period climatology, for every year.
3. Standardisation: one mean and standard deviation per band, pooled over all training
   anomalies, applied to every split; values are clipped to +/- clip.

Writes {output_dir}/{split}/{LOC}/{LOC}_{scaled,anomalies,doy,actual_doy,years}.npy and
{output_dir}/denorm.pkl with the climatology and the pooled statistics.

    python preprocessing/standardize.py --composites work/composites --output_dir work/zscore
"""

import argparse
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import rasterio

CHANNELS = ['NDVI', 'NIRv', 'sm_surface', 'sm_rootzone']


def load_composites(loc_dir: Path):
    files = sorted(loc_dir.glob('*.tif'))
    dates = [date.fromisoformat(f.stem) for f in files]
    data = []
    for f in files:
        with rasterio.open(f) as src:
            data.append(src.read())
    return dates, np.stack(data).astype(np.float32)


def period_index(d: date, interval: int, n_periods: int) -> int:
    return min((d.timetuple().tm_yday - 1) // interval, n_periods - 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--composites', required=True, help='output of composite.py')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--train_years', type=int, nargs=2, default=[2016, 2022])
    p.add_argument('--val_years', type=int, nargs=2, default=[2023, 2023])
    p.add_argument('--test_years', type=int, nargs=2, default=[2024, 2025])
    p.add_argument('--interval', type=int, default=8)
    p.add_argument('--clip', type=float, default=5.0)
    args = p.parse_args()

    splits = {name: range(lo, hi + 1) for name, (lo, hi) in
              (('train', args.train_years), ('val', args.val_years), ('test', args.test_years))}
    n_periods = -(-366 // args.interval)
    out = Path(args.output_dir)
    locations = sorted(d for d in Path(args.composites).iterdir() if d.is_dir())

    climatology = {}
    total = np.zeros(len(CHANNELS))
    total_sq = np.zeros(len(CHANNELS))
    count = np.zeros(len(CHANNELS))

    for loc_dir in locations:
        loc = loc_dir.name
        dates, data = load_composites(loc_dir)
        periods = np.array([period_index(d, args.interval, n_periods) for d in dates])
        years = np.array([d.year for d in dates])
        actual_doy = np.array([d.timetuple().tm_yday for d in dates])

        train = np.isin(years, splits['train'])
        climatology[loc] = {int(k): np.nanmean(data[train & (periods == k)], axis=0)
                            for k in np.unique(periods[train])}
        anomalies = np.stack([data[i] - climatology[loc][periods[i]] for i in range(len(data))])

        for image in anomalies[train]:
            for c, band in enumerate(image):
                values = band[~np.isnan(band)]
                total[c] += values.sum()
                total_sq[c] += (values ** 2).sum()
                count[c] += values.size

        for split, split_years in splits.items():
            keep = np.isin(years, split_years)
            if not keep.any():
                continue
            d = out / split / loc
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / f'{loc}_anomalies.npy', anomalies[keep].astype(np.float32))
            np.save(d / f'{loc}_doy.npy', periods[keep])
            np.save(d / f'{loc}_actual_doy.npy', actual_doy[keep])
            np.save(d / f'{loc}_years.npy', years[keep])

    mean = total / count
    std = np.sqrt(total_sq / count - mean ** 2)
    for split in splits:
        for loc_dir in sorted((out / split).glob('*')):
            loc = loc_dir.name
            anomalies = np.load(loc_dir / f'{loc}_anomalies.npy')
            scaled = (anomalies - mean[None, :, None, None]) / std[None, :, None, None]
            np.save(loc_dir / f'{loc}_scaled.npy', np.clip(scaled, -args.clip, args.clip).astype(np.float32))

    with open(out / 'denorm.pkl', 'wb') as f:
        pickle.dump({'climatology': climatology, 'global_mean': mean, 'global_std': std,
                     'channels': CHANNELS, 'clip_range': args.clip,
                     'composite_interval': args.interval,
                     **{f'{s}_years': list(y) for s, y in splits.items()}}, f)
    for c, name in enumerate(CHANNELS):
        print(f'{name:12s} mean {mean[c]:+.3e}  std {std[c]:.5f}')


if __name__ == '__main__':
    main()
