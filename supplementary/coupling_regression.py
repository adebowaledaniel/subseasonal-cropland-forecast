"""
Does the per-pixel RZSM skill gain follow coupling strength and aridity?

delta-ACC = ACC[NIRv+RZSM] - ACC[NIRv-only] per cropland pixel (per-pixel ACC as in
analysis/skill_maps.py, mean over the five Huber T=6 seeds) is related to the coupling
strength gamma and to a P/ET0 proxy (midpoint of the Global-AI v3 class). Significance
accounts for spatial clustering with patch-clustered standard errors and a patch block
bootstrap; Moran's I describes the clustering of delta-ACC within patches.

    python supplementary/coupling_regression.py --runs runs --data_root work/zscore \
        --cropland_masks work/cropland_masks --coupling figures/coupling_4326.tif \
        --aridity_classes work/aridity_classes_9km.tif --outdir figures
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import statsmodels.api as sm
from pyproj import Transformer
from scipy import sparse
from scipy.stats import pearsonr, t as t_dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))

from common import load_model  # noqa: E402
from skill_maps import location_acc_maps  # noqa: E402

SEEDS = [42, 123, 2024, 7, 99]
RUNS = {'rzsm': 'huber/nirv_rzsm_t6', 'only': 'huber/nirv_only_t6'}
CLASS_MIDPOINT = {1: 0.015, 2: 0.115, 3: 0.350, 4: 0.575, 5: 0.800}  # P/ET0 range midpoints
NODATA = -9999.0


def sample(path, x, y, band=1, transformer=None):
    """Nearest-pixel values of a raster at map coordinates (x, y); NaN outside or nodata."""
    if transformer is not None:
        x, y = transformer.transform(x, y)
    with rasterio.open(path) as src:
        values = src.read(band).astype(float)
        cols, rows = ~src.transform * (x, y)
        rows, cols = np.round(rows).astype(int), np.round(cols).astype(int)
        inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        out = values[np.clip(rows, 0, src.height - 1), np.clip(cols, 0, src.width - 1)]
        out = np.where(inside, out, np.nan)
        if src.nodata is not None:
            out = np.where(out == src.nodata, np.nan, out)
    return np.where(out == NODATA, np.nan, out)


def pixel_table(acc, args):
    to_lonlat = Transformer.from_crs('EPSG:6933', 'EPSG:4326', always_xy=True)
    rows = []
    for loc in sorted(set.intersection(*(set(m) for name in acc for m in acc[name].values()))):
        delta = np.nanmean([acc['rzsm'][s][loc] - acc['only'][s][loc] for s in SEEDS], axis=0)
        with rasterio.open(sorted((Path(args.cropland_masks) / loc).glob('cropland_percent_*.tif'))[-1]) as src:
            cropland = src.read(1) > 0
            r, c = np.indices(cropland.shape)
            x, y = (np.array(v).reshape(cropland.shape) for v in
                    rasterio.transform.xy(src.transform, r.ravel(), c.ravel(), offset='center'))
        gamma = sample(args.coupling, x, y, band=2, transformer=to_lonlat)
        classes = np.round(sample(args.aridity_classes, x, y))
        aridity = np.vectorize(lambda v: CLASS_MIDPOINT.get(int(v), np.nan) if np.isfinite(v) else np.nan)(classes)
        binary = sample(Path(args.aridity_dir) / loc / 'binary_ai.tif', x, y)
        keep = cropland & np.isfinite(delta) & np.isfinite(gamma) & np.isfinite(aridity) & np.isfinite(binary)
        for i, j in zip(*np.where(keep)):
            rows.append({'patch': loc, 'row': i, 'col': j, 'delta_acc': delta[i, j],
                         'gamma': gamma[i, j], 'aridity': aridity[i, j]})
    return pd.DataFrame(rows)


def partial_correlation(x, y, z):
    r_xy, r_xz, r_yz = pearsonr(x, y)[0], pearsonr(x, z)[0], pearsonr(y, z)[0]
    r = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    dof = len(x) - 3
    return r, 2 * (1 - t_dist.cdf(abs(r) * np.sqrt(dof / (1 - r ** 2)), dof))


def clustered_ols(df, columns):
    fit = sm.OLS(df['delta_acc'], sm.add_constant(df[columns])).fit(
        cov_type='cluster', cov_kwds={'groups': df['patch']})
    return {c: {'slope': fit.params[c], 'p': fit.pvalues[c]} for c in columns}


def block_bootstrap(df, column, rng, n_boot=2000):
    groups = {p: g for p, g in df.groupby('patch')}
    patches = list(groups)
    slopes = []
    for _ in range(n_boot):
        boot = pd.concat([groups[p] for p in rng.choice(patches, size=len(patches), replace=True)])
        if boot[column].std() > 0:
            slopes.append(np.polyfit(boot[column], boot['delta_acc'], 1)[0])
    slopes = np.array(slopes)
    return np.percentile(slopes, [2.5, 97.5]), min(1.0, 2 * min((slopes <= 0).mean(), (slopes >= 0).mean()))


def morans_i(df, rng, n_perm=999):
    """Global Moran's I with queen contiguity inside each patch, permutation p-value."""
    index = {(p, r, c): k for k, (p, r, c) in enumerate(zip(df['patch'], df['row'], df['col']))}
    pairs = [(k, index[p, r + dr, c + dc]) for (p, r, c), k in index.items()
             for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc) and (p, r + dr, c + dc) in index]
    i, j = zip(*pairs)
    weights = sparse.csr_matrix((np.ones(len(i)), (i, j)), shape=(len(df), len(df)))
    x = df['delta_acc'].values - df['delta_acc'].mean()
    stat = lambda v: len(v) / weights.sum() * (v @ (weights @ v)) / np.sum(v ** 2)  # noqa: E731
    observed = stat(x)
    null = np.array([stat(rng.permutation(x)) for _ in range(n_perm)])
    return observed, (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)


def statistics(df):
    df = df.assign(dryness=-df['aridity'])
    rng = np.random.default_rng(42)
    results = {'pixels': len(df), 'patches': df['patch'].nunique(),
               'pearson_r_gamma': pearsonr(df['gamma'], df['delta_acc'])[0],
               'pearson_r_gamma_aridity': pearsonr(df['gamma'], df['aridity'])[0],
               'partial_r_gamma_given_aridity': partial_correlation(df['gamma'], df['delta_acc'], df['dryness'])[0],
               'clustered_ols_gamma': clustered_ols(df, ['gamma'])['gamma'],
               'clustered_ols_dryness': clustered_ols(df, ['dryness'])['dryness'],
               'clustered_ols_both': clustered_ols(df, ['gamma', 'dryness'])}
    for column in ('gamma', 'dryness'):
        ci, p_boot = block_bootstrap(df, column, rng)
        results[f'block_bootstrap_{column}'] = {'ci95': ci.tolist(), 'p': p_boot}
    results['morans_i'], results['morans_i_p'] = morans_i(df, rng)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', default='runs')
    p.add_argument('--data_root', required=True)
    p.add_argument('--cropland_masks', required=True)
    p.add_argument('--coupling', required=True, help='coupling_4326.tif from analysis/coupling_maps.py')
    p.add_argument('--aridity_classes', required=True, help='Global-AI v3 classes (1-5) on the 9 km grid')
    p.add_argument('--aridity_dir', required=True)
    p.add_argument('--growing_season', default=str(ROOT / 'configs/growing_season.yml'))
    p.add_argument('--outdir', default='figures')
    args = p.parse_args()

    acc = {name: {} for name in RUNS}
    for name, run in RUNS.items():
        for seed in SEEDS:
            model, config = load_model(Path(args.runs) / run, seed)
            acc[name][seed] = location_acc_maps(model, config, Path(args.data_root) / 'test', args.growing_season)
    results = statistics(pixel_table(acc, args))

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.outdir) / 'coupling_regression.json', 'w') as f:
        json.dump(results, f, indent=1, default=float)
    print(json.dumps(results, indent=1, default=float))


if __name__ == '__main__':
    main()
