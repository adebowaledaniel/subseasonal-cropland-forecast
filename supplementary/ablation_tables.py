"""
Input-window ablations.

history:      water-limited ACC (mean over leads and seeds) of the dual-encoder runs with
              separate NIRv and RZSM history lengths (configs/ablation/history_a*.yaml).
day of year:  water-limited delta-ACC (NIRv+RZSM - NIRv-only, mean over leads) without and
              with sin/cos day-of-year inputs, at T=6 and T=10.
--peak_season: the same delta-ACC on forecasts whose target month lies in the peak growing
              season (GEOGLAM vegetative to end of season) of their location.

    python supplementary/ablation_tables.py --runs runs --data_root work/zscore --peak_season
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
sys.path.insert(0, str(ROOT / 'preprocessing'))

from common import load_model  # noqa: E402
from crop_calendar import season_months  # noqa: E402
from src.data import month_of  # noqa: E402
from src.evaluation import predict  # noqa: E402
from src.metrics import acc  # noqa: E402
from src.training import build_dataset, make_loader  # noqa: E402

SEEDS = [42, 123, 2024, 7, 99]
HISTORY = {'A4': 'history_a4', 'A1': 'history_a1', 'A2': 'history_a2', 'A3': 'history_a3'}
DOY_RUNS = {False: ('huber/nirv_rzsm', 'huber/nirv_only'),
            True: ('ablation/doy_nirv_rzsm', 'ablation/doy_nirv_only')}


def runs(root, run):
    return [json.load(open(f)) for f in sorted((Path(root) / run).glob('seed_*/test_metrics.json'))]


def mean_over_leads(results, key='water_limited_acc_lead'):
    return np.mean([np.mean(r[key]) for r in results])


def history_table(root):
    rows = []
    for name, run in HISTORY.items():
        results = runs(root, f'ablation/{run}')
        rows.append({'config': name, 'WL ACC': mean_over_leads(results), 'parameters': results[0]['parameters']})
    return pd.DataFrame(rows)


def day_of_year_table(root):
    rows = []
    for tlen in (6, 10):
        row = {'T': tlen}
        for doy, (rzsm, only) in DOY_RUNS.items():
            delta = mean_over_leads(runs(root, f'{rzsm}_t{tlen}')) - mean_over_leads(runs(root, f'{only}_t{tlen}'))
            row['dWL ACC (+DOY)' if doy else 'dWL ACC (no DOY)'] = delta
        rows.append(row)
    return pd.DataFrame(rows)


def peak_months(attributes):
    out = {}
    with open(attributes) as f:
        for row in csv.DictReader(f):
            out[row['Name']] = (set(season_months(row['vegetative'], row['harvest']))
                                | set(season_months(row['vegetative'], row['endofseaso']))
                                | set(season_months(row['harvest'], row['endofseaso'])))
    return out


def peak_season_acc(model, config, args, peak):
    dataset = build_dataset(config, Path(args.data_root) / 'test', args.growing_season)
    out = predict(model, make_loader(dataset, dict(config, num_workers=0)), 'cpu')
    values = []
    for lead in range(out['pred'].shape[1]):
        keep = np.array([month_of(d, y) in peak.get(loc, set()) for loc, d, y in
                         zip(out['location'], out['target_doy'][:, lead], out['target_year'][:, lead])])
        values.append(acc(out['pred'][keep, lead], out['obs'][keep, lead]) if keep.any() else np.nan)
    return np.array(values)


def table_peak_season(args):
    peak = peak_months(ROOT / 'data/grid_attributes.csv')
    rows = []
    for tlen in (6, 10):
        row = {'T': tlen}
        for doy, (rzsm, only) in DOY_RUNS.items():
            mean = {}
            for run in (rzsm, only):
                per_seed = [peak_season_acc(*load_model(Path(args.runs) / f'{run}_t{tlen}', s), args, peak)
                            for s in SEEDS]
                mean[run] = np.mean(per_seed, axis=0)
            row['dACC peak season (+DOY)' if doy else 'dACC peak season (no DOY)'] = np.mean(mean[rzsm] - mean[only])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='runs')
    p.add_argument('--data_root')
    p.add_argument('--growing_season', default=str(ROOT / 'configs/growing_season.yml'))
    p.add_argument('--peak_season', action='store_true')
    p.add_argument('--outdir', default='figures')
    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tables = {'history_length': history_table(args.runs), 'day_of_year': day_of_year_table(args.runs)}
    if args.peak_season:
        tables['peak_season_day_of_year'] = table_peak_season(args)
    for name, table in tables.items():
        print(f'{name}\n{table.round(3).to_string(index=False)}\n')
        table.to_csv(outdir / f'{name}.csv', index=False)


if __name__ == '__main__':
    main()
