"""
Leave-one-country-out (LOCO) generalisation.

build: writes a data root without the held-out country in train/ and val/ and with only
       that country in test/. The per-location climatology is unchanged; the pooled
       per-band mean and SD are recomputed from the remaining training locations and every
       location is re-standardised with them.
table: delta-ACC (NIRv+RZSM - NIRv-only, mean over leads and seeds) of the LOCO models
       against the all-patches models evaluated on the same country's locations.

    python supplementary/loco.py build --data_root work/zscore --country ZWE --output work/zscore_loco_ZWE
    python supplementary/loco.py table --runs runs --data_root work/zscore
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))

from common import load_model  # noqa: E402
from src.evaluation import predict  # noqa: E402
from src.metrics import acc  # noqa: E402
from src.training import build_dataset, make_loader  # noqa: E402

COUNTRIES = ['TZA', 'ZWE']
TLENS = [6, 10]
SEEDS = [42, 123, 2024, 7, 99]
CONFIGS = ('nirv_rzsm', 'nirv_only')


def build(data_root: Path, country: str, output: Path, clip: float = 5.0):
    train = sorted(d.name for d in (data_root / 'train').iterdir() if d.is_dir())
    held_out = [loc for loc in train if loc.startswith(f'{country}_')]
    kept = [loc for loc in train if loc not in held_out]

    total, total_sq, count = np.zeros(4), np.zeros(4), np.zeros(4)
    for loc in kept:
        anomalies = np.load(data_root / 'train' / loc / f'{loc}_anomalies.npy')
        for c in range(4):
            values = anomalies[:, c][~np.isnan(anomalies[:, c])]
            total[c] += values.sum()
            total_sq[c] += (values ** 2).sum()
            count[c] += values.size
    mean = total / count
    std = np.sqrt(total_sq / count - mean ** 2)

    for split, locations in (('train', kept), ('val', kept), ('test', held_out)):
        for loc in locations:
            src, dst = data_root / split / loc, output / split / loc
            if not src.is_dir():
                continue
            dst.mkdir(parents=True, exist_ok=True)
            anomalies = np.load(src / f'{loc}_anomalies.npy')
            scaled = (anomalies - mean[None, :, None, None]) / std[None, :, None, None]
            np.save(dst / f'{loc}_scaled.npy', np.clip(scaled, -clip, clip).astype(np.float32))
            for f in src.glob(f'{loc}_*.npy'):
                if not f.name.endswith('_scaled.npy') and not (dst / f.name).exists():
                    (dst / f.name).symlink_to(f.resolve())
    print(f'{country}: {len(held_out)} locations held out, {len(kept)} kept')


def all_patches_acc(runs_root, data_root, growing_season, config_name, tlen, country):
    """Per-lead ACC of the all-patches model, averaged over the country's locations and seeds."""
    values = []
    for seed in SEEDS:
        model, config = load_model(Path(runs_root) / f'huber/{config_name}_t{tlen}', seed)
        locations = [d.name for d in sorted((data_root / 'test').iterdir()) if d.name.startswith(f'{country}_')]
        dataset = build_dataset(config, data_root / 'test', growing_season, locations=locations)
        out = predict(model, make_loader(dataset, dict(config, num_workers=0)), 'cpu')
        names = np.array(out['location'])
        for loc in dict.fromkeys(out['location']):
            idx = names == loc
            values.append([acc(out['pred'][idx, t], out['obs'][idx, t]) for t in range(out['pred'].shape[1])])
    return np.mean(values, axis=0)


def loco_acc(runs_root, config_name, tlen, country):
    files = sorted((Path(runs_root) / f'loco/{country}_{config_name}_t{tlen}').glob('seed_*/test_metrics.json'))
    return np.mean([json.load(open(f))['acc_lead'] for f in files], axis=0)


def table(args):
    rows = []
    for country in COUNTRIES:
        for tlen in TLENS:
            loco = {c: loco_acc(args.runs, c, tlen, country) for c in CONFIGS}
            full = {c: all_patches_acc(args.runs, Path(args.data_root), args.growing_season, c, tlen, country)
                    for c in CONFIGS}
            rows.append({'held-out country': country, 'T': tlen,
                         'dACC (LOCO)': np.mean(loco['nirv_rzsm'] - loco['nirv_only']),
                         'dACC (all-patches model)': np.mean(full['nirv_rzsm'] - full['nirv_only'])})
    result = pd.DataFrame(rows).round(3)
    print(result.to_string(index=False))
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    result.to_csv(Path(args.outdir) / 'loco.csv', index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--data_root', required=True)
    b.add_argument('--country', required=True, help='ISO3 prefix of the held-out cells')
    b.add_argument('--output', required=True)
    t = sub.add_parser('table')
    t.add_argument('--runs', default='runs')
    t.add_argument('--data_root', required=True)
    t.add_argument('--growing_season', default=str(ROOT / 'configs/growing_season.yml'))
    t.add_argument('--outdir', default='figures')
    args = p.parse_args()
    if args.command == 'build':
        build(Path(args.data_root), args.country, Path(args.output))
    else:
        table(args)


if __name__ == '__main__':
    main()
