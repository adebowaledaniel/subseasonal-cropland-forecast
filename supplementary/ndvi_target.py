"""
NDVI instead of NIRv as the forecast target (T=6, Huber runs).

Prints the RZSM contribution per lead for both targets (ACC[+RZSM] - ACC[target only]) and
the mean ACC and skill score over the three configurations and five leads, where the
per-lead skill score is 1 - RMSE / climatology RMSE on observed pixels.

    python supplementary/ndvi_target.py --runs runs --outdir figures
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = {'NIRv': 'nirv', 'NDVI': 'ndvi'}
CONFIGS = {'target only': 'only', 'target+RZSM': 'rzsm', 'target+lagRZSM': 'lagrzsm'}


def load(runs_root, target, config):
    files = sorted((Path(runs_root) / f'huber/{target}_{config}_t6').glob('seed_*/test_metrics.json'))
    return [json.load(open(f)) for f in files]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='runs')
    p.add_argument('--outdir', default='figures')
    args = p.parse_args()

    rows = []
    for target, t in TARGETS.items():
        for config, c in CONFIGS.items():
            runs = load(args.runs, t, c)
            acc = np.array([r['acc_lead'] for r in runs])
            ss = 1 - np.array([r['rmse_lead'] for r in runs]) / np.array([r['clim_rmse_lead'] for r in runs])
            for lead in range(acc.shape[1]):
                rows.append({'target': target, 'config': config, 'lead': lead + 1,
                             'acc': acc[:, lead].mean(), 'acc_sd': acc[:, lead].std(),
                             'ss': ss[:, lead].mean(), 'ss_sd': ss[:, lead].std()})
    table = pd.DataFrame(rows)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(Path(args.outdir) / 'ndvi_vs_nirv.csv', index=False)

    acc = table.pivot_table(index='lead', columns=['target', 'config'], values='acc')
    delta = pd.DataFrame({target: acc[target, 'target+RZSM'] - acc[target, 'target only'] for target in TARGETS})
    print('RZSM contribution, delta-ACC (target+RZSM - target only):')
    print(delta.round(3).to_string())
    delta.round(4).to_csv(Path(args.outdir) / 'ndvi_rzsm_contribution.csv')
    print('\nMean over configurations and leads:')
    print(table.groupby('target')[['acc', 'ss']].mean().round(3).to_string())


if __name__ == '__main__':
    main()
