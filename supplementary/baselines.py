"""
The ConvLSTM against z-score persistence, the zero-anomaly
climatology, pooled per-pixel ridge regression, XGBoost and the non-spatial pixel LSTM, at
T=6, H=5, on the same test windows. Also prints the RZSM gain within each model class and
the ConvLSTM - LSTM difference paired by seed.

The ridge and XGBoost models are fitted per lead on every observed pixel of the training
windows; their hyperparameters are chosen on the validation year. The ConvLSTM and LSTM
rows are read from the train.py runs listed in RUNS.

    python supplementary/baselines.py --runs runs --data_root work/zscore --outdir figures
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.metrics import forecast_skill  # noqa: E402
from src.training import build_dataset  # noqa: E402

VARIANTS = {'NIRv-only': 'nirv_only', 'NIRv+RZSM': 'nirv_rzsm'}
RUNS = {'LSTM (non-spatial)': 'baselines/lstm_{}_t6', 'ConvLSTM': 'huber/{}_t6'}
SEEDS = [42, 123, 2024, 7, 99]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
XGB_GRID = list(product([2, 4], [100, 200], [0.1]))  # max_depth, n_estimators, learning_rate
XGB_MAX_TRAIN_ROWS = 60_000
N_PERIODS = 46


def split_arrays(config, split_dir, growing_season):
    dataset = build_dataset(config, split_dir, growing_season)
    samples = [dataset[i] for i in range(len(dataset))]
    stack = lambda key: np.stack([s[key].numpy() for s in samples])  # noqa: E731
    return {'inputs': stack('inputs'), 'target': stack('target')[:, :, 0],
            'mask': stack('target_mask')[:, :, 0], 'period': stack('target_period')}


def pixel_rows(arrays, lead, day_of_year=False):
    """Features (N*H*W, T*C [+2]), targets and observed flags for one lead."""
    n, t, c, h, w = arrays['inputs'].shape
    x = arrays['inputs'].transpose(0, 3, 4, 1, 2).reshape(n * h * w, t * c)
    if day_of_year:
        angle = np.repeat(2 * np.pi * arrays['period'][:, lead].astype(np.float32) / N_PERIODS, h * w)
        x = np.concatenate([x, np.sin(angle)[:, None], np.cos(angle)[:, None]], axis=1)
    return x, arrays['target'][:, lead].reshape(-1), arrays['mask'][:, lead].reshape(-1).astype(bool)


def val_rmse(model, x, y):
    return float(np.sqrt(np.mean((model.predict(x) - y) ** 2)))


def ridge_forecast(train, val, test):
    pred = np.zeros_like(test['target'])
    for lead in range(pred.shape[1]):
        xt, yt, mt = pixel_rows(train, lead)
        xv, yv, mv = pixel_rows(val, lead)
        models = [Ridge(alpha=a).fit(xt[mt], yt[mt]) for a in RIDGE_ALPHAS]
        best = min(models, key=lambda m: val_rmse(m, xv[mv], yv[mv]))
        pred[:, lead] = best.predict(pixel_rows(test, lead)[0]).reshape(pred[:, lead].shape)
    return pred


def fit_xgb(x, y, max_depth, n_estimators, lr, seed):
    model = xgb.XGBRegressor(max_depth=max_depth, n_estimators=n_estimators, learning_rate=lr,
                             subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                             random_state=seed, n_jobs=1, objective='reg:squarederror')
    return model.fit(x, y)


def xgb_forecasts(train, val, test):
    preds = {seed: np.zeros_like(test['target']) for seed in SEEDS}
    for lead in range(test['target'].shape[1]):
        xt, yt, mt = pixel_rows(train, lead, day_of_year=True)
        xv, yv, mv = pixel_rows(val, lead, day_of_year=True)
        xt, yt = xt[mt], yt[mt]
        if len(yt) > XGB_MAX_TRAIN_ROWS:
            keep = np.random.RandomState(42).choice(len(yt), XGB_MAX_TRAIN_ROWS, replace=False)
            xt, yt = xt[keep], yt[keep]
        params = min(XGB_GRID, key=lambda p: val_rmse(fit_xgb(xt, yt, *p, seed=42), xv[mv], yv[mv]))
        x_test = pixel_rows(test, lead, day_of_year=True)[0]
        shape = test['target'][:, lead].shape
        for seed in SEEDS:
            preds[seed][:, lead] = fit_xgb(xt, yt, *params, seed).predict(x_test).reshape(shape)
    return preds


def score(pred, test, seed=None):
    m = forecast_skill(pred, test['target'], test['mask'], mask_regression=True)
    return dict(m, seed=seed)


def paired_difference(a, b, key='acc'):
    """Mean and SD (ddof=0) of a - b over the seeds common to both lists of runs."""
    a, b = {r['seed']: r[key] for r in a}, {r['seed']: r[key] for r in b}
    diffs = np.array([a[s] - b[s] for s in sorted(set(a) & set(b))])
    return diffs.mean(), diffs.std()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='runs')
    p.add_argument('--data_root', required=True)
    p.add_argument('--growing_season', default=str(ROOT / 'configs/growing_season.yml'))
    p.add_argument('--outdir', default='figures')
    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root)

    results = {}
    for variant, name in VARIANTS.items():
        config = load_config(ROOT / f'configs/{name}.yaml')
        train, val, test = (split_arrays(config, root / s, args.growing_season) for s in ('train', 'val', 'test'))
        if variant == 'NIRv-only':
            persistence = np.repeat(test['inputs'][:, -1:, 0], test['target'].shape[1], axis=1)
            results['Persistence', 'NIRv'] = [score(persistence, test)]
            climatology = score(np.zeros_like(test['target']), test)
            climatology.update(acc=0.0, acc_lead=[0.0] * len(climatology['acc_lead']))
            results['Climatology', 'zero-anomaly'] = [climatology]
        results['Ridge regression', variant] = [score(ridge_forecast(train, val, test), test)]
        xgb_preds = xgb_forecasts(train, val, test)
        results['XGBoost', variant] = [score(pred, test, seed) for seed, pred in xgb_preds.items()]
        for model, run in RUNS.items():
            files = sorted((Path(args.runs) / run.format(name)).glob('seed_*/test_metrics.json'))
            results[model, variant] = [json.load(open(f)) for f in files]

    rows = [{'Model': model, 'Input': variant, 'n': len(runs),
             **{k.upper(): np.mean([r[k] for r in runs]) for k in ('acc', 'rmse')},
             'SS': np.mean([r['skill_score'] for r in runs])}
            for (model, variant), runs in results.items()]
    table = pd.DataFrame(rows).round(3)
    print(table.to_string(index=False))
    table.to_csv(outdir / 'baselines.csv', index=False)

    print('\nRZSM gain in domain ACC (NIRv+RZSM - NIRv-only):')
    for model in ('Ridge regression', 'XGBoost', 'LSTM (non-spatial)', 'ConvLSTM'):
        gain = (np.mean([r['acc'] for r in results[model, 'NIRv+RZSM']])
                - np.mean([r['acc'] for r in results[model, 'NIRv-only']]))
        print(f'  {model:20s} {gain:+.3f}')
    mean, sd = paired_difference(results['ConvLSTM', 'NIRv+RZSM'], results['LSTM (non-spatial)', 'NIRv+RZSM'])
    print(f'ConvLSTM - LSTM (NIRv+RZSM), paired by seed: {mean:+.3f} +/- {sd:.3f}')


if __name__ == '__main__':
    main()
