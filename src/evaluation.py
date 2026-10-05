"""Inference, aridity lookup and test-set evaluation."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio
import torch

from .data import model_kwargs
from .metrics import forecast_skill


@torch.no_grad()
def predict(model: torch.nn.Module, loader, device) -> Dict:
    """
    Run `model` over `loader`. Returns arrays of shape (N, horizon, H, W) for 'pred',
    'obs' and 'mask', and per-sample 'location', 'target_doy' and 'target_year'.
    """
    model.eval()
    out = {k: [] for k in ('pred', 'obs', 'mask', 'location', 'target_doy', 'target_year')}
    for batch in loader:
        pred = model(batch['inputs'].to(device), **model_kwargs(batch, device))
        out['pred'].append(pred[:, :, 0].cpu().numpy())
        out['obs'].append(batch['target'][:, :, 0].numpy())
        out['mask'].append(batch['target_mask'][:, :, 0].numpy())
        out['location'].extend(batch['location'])
        out['target_doy'].append(batch['target_doy'].numpy())
        out['target_year'].append(batch['target_year'].numpy())
    for key in ('pred', 'obs', 'mask', 'target_doy', 'target_year'):
        out[key] = np.concatenate(out[key])
    return out


def load_aridity(aridity_dir, locations: List[str]) -> np.ndarray:
    """
    Aridity class of each sample's patch, read from {aridity_dir}/{LOC}/binary_ai.tif
    (1 = water-limited, P/ET0 < 0.5; 2 = energy-limited). Missing patches are 0.
    """
    cache = {}
    for loc in set(locations):
        path = Path(aridity_dir) / loc / 'binary_ai.tif'
        if path.exists():
            with rasterio.open(path) as src:
                cache[loc] = src.read(1)
    if not cache:
        raise FileNotFoundError(f'no binary_ai.tif found under {aridity_dir}')
    shape = next(iter(cache.values())).shape
    return np.stack([cache.get(loc, np.zeros(shape)) for loc in locations])


def evaluate(model, loader, device, aridity_dir=None) -> Dict:
    """Skill of `model` on `loader` (see metrics.forecast_skill)."""
    results = predict(model, loader, device)
    aridity = load_aridity(aridity_dir, results['location']) if aridity_dir else None
    metrics = forecast_skill(results['pred'], results['obs'], results['mask'], aridity)
    metrics['n_samples'] = int(len(results['pred']))
    return metrics


def summarise(runs: List[Dict]) -> Dict:
    """Mean and standard deviation (ddof=0) across seeds of every numeric metric."""
    summary = {'seeds': [r['seed'] for r in runs]}
    for key, value in runs[0].items():
        if key == 'seed' or not isinstance(value, (int, float, list)):
            continue
        values = np.array([r[key] for r in runs], dtype=float)
        summary[key] = {'mean': np.nanmean(values, axis=0).tolist(),
                        'std': np.nanstd(values, axis=0).tolist()}
    return summary
