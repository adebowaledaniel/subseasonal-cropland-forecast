"""Experiment configuration: a flat dictionary read from YAML on top of these defaults."""

from pathlib import Path
from typing import Optional

import yaml

DEFAULTS = {
    # data
    'input_bands': [1, 3],          # 0 NDVI, 1 NIRv, 2 surface SM, 3 root-zone SM
    'target_band': 1,
    'lagged_rzsm': False,           # append RZSM shifted by the per-pixel optimal lag
    'lag_variable': 'NIRV',         # which {LOC}_{VAR}_sm_rootzone.npy lag map to use
    'tlen': 6,
    'forecast_horizon': 5,
    'growing_season_only': True,
    'min_valid_ratio': 0.8,
    'decoupled_history': None,      # [tlen_nirv, tlen_rzsm] for the dual-encoder ablation
    'day_of_year': False,           # add sin/cos day-of-year inputs (ablation)
    # model
    'model': 'convlstm',            # convlstm | pixel_lstm
    'hidden_dims': [64, 64],        # convlstm
    'kernel_sizes': [3, 3],         # convlstm
    'hidden_dim': 64,               # pixel_lstm
    'num_layers': 2,                # pixel_lstm
    'dropout': 0.18,
    # optimisation
    'loss': 'huber',                # huber | mse
    'huber_delta': 1.0,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'batch_size': 32,
    'epochs': 100,
    'grad_clip': 1.0,
    'lr_factor': 0.5,
    'lr_patience': 5,
    'patience': 25,
    'min_delta': 5e-5,
    'num_workers': 8,
}


def load_config(path: Optional[str] = None, **overrides) -> dict:
    config = dict(DEFAULTS)
    if path:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            raise KeyError(f'unknown config keys in {path}: {sorted(unknown)}')
        config.update(user)
    config.update({k: v for k, v in overrides.items() if v is not None})
    return config


def save_config(config: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)
