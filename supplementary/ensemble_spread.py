"""
Calibration of the five-seed ensemble spread.

The NIRv+RZSM seeds (Huber, T=6) forecast every growing-season test window. Per pixel and
lead, the nominal 95% interval is the ensemble mean +/- 1.96 SD (ddof=1), and the empirical
coverage is the fraction of cropland pixels whose observation falls inside it. Cropland
pixels without an observation count as outside the interval.

    python supplementary/ensemble_spread.py --runs runs --data_root work/zscore \
        --cropland_masks work/cropland_masks --outdir figures
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))

from common import load_model  # noqa: E402
from src.evaluation import predict  # noqa: E402
from src.training import build_dataset, make_loader  # noqa: E402

SEEDS = [42, 123, 2024, 7, 99]
RUN = 'huber/nirv_rzsm_t6'
Z95 = 1.959963985


def cropland(cropland_masks, location):
    with rasterio.open(sorted((Path(cropland_masks) / location).glob('cropland_percent_*.tif'))[-1]) as src:
        return src.read(1) > 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='runs')
    p.add_argument('--data_root', required=True)
    p.add_argument('--cropland_masks', required=True)
    p.add_argument('--growing_season', default=str(ROOT / 'configs/growing_season.yml'))
    p.add_argument('--outdir', default='figures')
    args = p.parse_args()

    preds, out = [], None
    for seed in SEEDS:
        model, config = load_model(Path(args.runs) / RUN, seed)
        dataset = build_dataset(config, Path(args.data_root) / 'test', args.growing_season,
                                season_filter='target', min_valid_ratio=0.0)
        out = predict(model, make_loader(dataset, dict(config, num_workers=0)), 'cpu')
        preds.append(out['pred'])
    preds = np.stack(preds)
    obs = np.where(out['mask'], out['obs'], np.nan)
    masks = {loc: cropland(args.cropland_masks, loc) for loc in set(out['location'])}
    crop = np.stack([masks[loc] for loc in out['location']])[:, None].repeat(obs.shape[1], axis=1)

    mean, sd = preds.mean(axis=0), preds.std(axis=0, ddof=1)
    inside = (obs >= mean - Z95 * sd) & (obs <= mean + Z95 * sd)
    rows = [{'lead': lead + 1, 'nominal': 0.95,
             'empirical': inside[:, lead][crop[:, lead]].mean(),
             'mean_ensemble_sd': sd[:, lead][crop[:, lead]].mean(),
             'n': int(crop[:, lead].sum())} for lead in range(obs.shape[1])]
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(args.outdir) / 'ensemble_coverage.csv', index=False)


if __name__ == '__main__':
    main()
