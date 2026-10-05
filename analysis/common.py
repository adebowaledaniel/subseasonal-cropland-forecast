"""Shared constants and helpers for the figure and table scripts."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.io import MemoryFile
from rasterio.merge import merge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.models import build_model  # noqa: E402

LEADS = np.arange(1, 6)
LEAD_DAYS = LEADS * 8
LEAD_LABELS = [f't+{lead}\n({days}d)' for lead, days in zip(LEADS, LEAD_DAYS)]
TLENS = [2, 4, 6, 8, 10, 12]
CONFIGS = ['NIRv+RZSM', 'NIRv+lagRZSM', 'NIRv-only']

DOMAIN_RUNS = {'NIRv+RZSM': 'huber/nirv_rzsm', 'NIRv+lagRZSM': 'mse/nirv_lagrzsm',
               'NIRv-only': 'mse/nirv_only'}
LEAD_FIGURE_RUNS = {'NIRv+RZSM': 'mse/nirv_rzsm', 'NIRv+lagRZSM': 'mse/nirv_lagrzsm',
                    'NIRv-only': 'mse/nirv_only'}
LEAD_FIGURE_SEEDS = (42, 123, 2024)
REGIME_RUNS = {'NIRv+RZSM': 'huber/nirv_rzsm', 'NIRv+lagRZSM': 'huber/nirv_lagrzsm',
               'NIRv-only': 'huber/nirv_only'}

COLORS = {'NIRv+RZSM': '#2166AC', 'NIRv+lagRZSM': '#4DAF4A', 'NIRv-only': '#B2182B'}
MARKERS = {'NIRv+RZSM': 'o', 'NIRv+lagRZSM': '^', 'NIRv-only': 's'}


def load_runs(runs_root, run: str, tlen: int, seeds=None):
    """Per-seed test metrics of one experiment, e.g. load_runs('runs', 'huber/nirv_rzsm', 6)."""
    files = sorted((Path(runs_root) / f'{run}_t{tlen}').glob('seed_*/test_metrics.json'))
    runs = [json.load(open(f)) for f in files]
    if seeds is not None:
        runs = [r for r in runs if r['seed'] in seeds]
    if not runs:
        raise FileNotFoundError(f'no test metrics under {runs_root}/{run}_t{tlen}')
    return runs


def seed_stats(runs, key: str):
    """Mean and standard deviation (ddof=0) across seeds of a scalar or per-lead metric."""
    values = np.array([r[key] for r in runs], dtype=float)
    return values.mean(axis=0), values.std(axis=0)


def save(fig, outdir, name: str) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(outdir / f'{name}.{ext}', dpi=350, bbox_inches='tight')
    plt.close(fig)


def load_model(run_dir, seed: int = 42, device='cpu'):
    """Model and config of one seed of a train.py run."""
    config = load_config(Path(run_dir) / 'config.yaml')
    model = build_model(config)
    state = torch.load(Path(run_dir) / f'seed_{seed}' / 'best.pt', map_location=device)
    model.load_state_dict(state['model'])
    return model.to(device).eval(), config


def mosaic(maps, cropland_masks):
    """
    Merge per-location (H, W) arrays on the 9 km grid, masked to cropland of the latest
    cropland map. Returns the mosaic and its transform (EPSG:6933).
    """
    files = []
    for loc, values in maps.items():
        latest = sorted((Path(cropland_masks) / loc).glob('cropland_percent_*.tif'))[-1]
        with rasterio.open(latest) as src:
            masked = np.where(src.read(1) > 0, values, np.nan).astype(np.float32)
            profile = dict(driver='GTiff', dtype='float32', count=1, height=src.height,
                           width=src.width, crs=src.crs, transform=src.transform, nodata=np.nan)
        memfile = MemoryFile()
        with memfile.open(**profile) as dst:
            dst.write(masked, 1)
        files.append(memfile)
    array, transform = merge([f.open() for f in files])
    return array[0], transform
