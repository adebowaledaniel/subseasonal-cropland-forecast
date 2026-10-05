"""
Sliding-window datasets over the preprocessed patch time series.

Each location directory of a split holds
    {LOC}_scaled.npy      (T, 4, 32, 32) standardised DOY anomalies, NaN outside cropland
    {LOC}_doy.npy         (T,) 8-day composite index (0-45)
    {LOC}_actual_doy.npy  (T,) calendar day of year of each composite
    {LOC}_years.npy       (T,) calendar year of each composite
    {LOC}_{VAR}_sm_rootzone.npy  (32, 32) optimal RZSM lag in composites (lagged input only)
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

BANDS = {'NDVI': 0, 'NIRv': 1, 'SSM': 2, 'RZSM': 3}


def load_growing_season(path) -> Dict[str, List[int]]:
    """Growing-season months per location, e.g. {'ZMB_G_12': [11, 12, 1, 2, 3]}."""
    with open(path) as f:
        return yaml.safe_load(f)


def month_of(day_of_year: int, year: int) -> int:
    return (date(int(year), 1, 1) + timedelta(days=int(day_of_year) - 1)).month


class PatchSequenceDataset(Dataset):
    """
    Samples (tlen inputs -> forecast_horizon targets) from every location of a split.

    Windows are kept when at least `min_valid_ratio` of the input (all four bands) and
    of the target values are observed. With `growing_season`, windows must also lie in
    the location's growing season: every input composite for season_filter='input'
    (training), every target composite for season_filter='target' (the
    per-pixel skill maps).

    With lagged_rzsm=True an extra channel is appended whose value at pixel (h, w) and
    step t is RZSM at t - L_opt(h, w), using the lag map named by lag_variable. Pixels
    without a significant lag use concurrent RZSM (missing_lag='concurrent', as in
    training) or zero (missing_lag='zero').

    split_dir may also be a list of split directories, whose series are joined in order
    (e.g. [val, test] so that forecasts early in the test period have full input windows).
    """

    def __init__(self, split_dir, input_bands: Sequence[int] = (1, 3), target_band: int = 1,
                 tlen: int = 6, forecast_horizon: int = 5, lagged_rzsm: bool = False,
                 lag_variable: str = 'NIRV', growing_season: Optional[Dict[str, List[int]]] = None,
                 season_filter: str = 'input', min_valid_ratio: float = 0.8,
                 locations: Optional[Sequence[str]] = None, missing_lag: str = 'concurrent'):
        dirs = [split_dir] if isinstance(split_dir, (str, Path)) else split_dir
        self.split_dirs = [Path(d) for d in dirs]
        self.input_bands = list(input_bands)
        self.target_band = target_band
        self.tlen = tlen
        self.forecast_horizon = forecast_horizon
        self.lagged_rzsm = lagged_rzsm
        self.lag_variable = lag_variable
        self.growing_season = growing_season
        self.season_filter = season_filter
        self.min_valid_ratio = min_valid_ratio
        self.missing_lag = missing_lag

        if locations is None:
            locations = sorted(d.name for d in self.split_dirs[-1].iterdir() if d.is_dir())
        self.locations = list(locations)
        self.series = {loc: self._load(loc) for loc in self.locations}
        self.index = [(loc, start) for loc in self.locations for start in self._windows(loc)]

    def _load(self, loc: str) -> Dict[str, np.ndarray]:
        dirs = [s / loc for s in self.split_dirs if (s / loc).is_dir()]
        series = {key: np.concatenate([np.load(d / f'{loc}_{key}.npy') for d in dirs])
                  for key in ('scaled', 'doy', 'actual_doy', 'years')}
        if self.lagged_rzsm:
            lag_file = dirs[-1] / f'{loc}_{self.lag_variable}_sm_rootzone.npy'
            lag = np.load(lag_file) if lag_file.exists() else np.zeros(series['scaled'].shape[2:])
            series['lag_missing'] = np.isnan(lag) | (lag == -9999)
            series['lag'] = np.where(series['lag_missing'], 0, lag).astype(int)
        return series

    def _in_season(self, loc: str, steps: np.ndarray) -> bool:
        months = self.growing_season.get(loc, [])
        s = self.series[loc]
        return all(month_of(day, year) in months
                   for day, year in zip(s['actual_doy'][steps], s['years'][steps]))

    def _windows(self, loc: str) -> List[int]:
        data = self.series[loc]['scaled']
        n = len(data) - self.tlen - self.forecast_horizon + 1
        starts = []
        for start in range(max(n, 0)):
            target_start = start + self.tlen
            inputs = data[start:target_start]
            target = data[target_start:target_start + self.forecast_horizon, self.target_band]
            if (np.mean(~np.isnan(inputs)) < self.min_valid_ratio
                    or np.mean(~np.isnan(target)) < self.min_valid_ratio):
                continue
            if self.growing_season is not None:
                steps = (np.arange(start, target_start) if self.season_filter == 'input'
                         else np.arange(target_start, target_start + self.forecast_horizon))
                if not self._in_season(loc, steps):
                    continue
            starts.append(start)
        return starts

    def lagged_channel(self, loc: str, start: int) -> np.ndarray:
        s = self.series[loc]
        rzsm = s['scaled'][:, BANDS['RZSM']]
        source = np.arange(start, start + self.tlen)[:, None, None] - s['lag'][None]
        valid = (source >= 0) & (source < len(rzsm))
        h, w = np.indices(s['lag'].shape)
        values = rzsm[np.clip(source, 0, len(rzsm) - 1), h[None], w[None]]
        if self.missing_lag == 'zero':
            valid &= ~s['lag_missing'][None]
        return np.where(valid, values, 0.0)[:, None]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        loc, start = self.index[idx]
        s = self.series[loc]
        target_start = start + self.tlen
        inp = slice(start, target_start)
        tgt = slice(target_start, target_start + self.forecast_horizon)

        inputs = s['scaled'][inp][:, self.input_bands]
        if self.lagged_rzsm:
            inputs = np.concatenate([inputs, self.lagged_channel(loc, start)], axis=1)
        target = s['scaled'][tgt][:, [self.target_band]]

        return {
            'inputs': torch.from_numpy(np.nan_to_num(inputs, nan=0.0)).float(),
            'target': torch.from_numpy(np.nan_to_num(target, nan=0.0)).float(),
            'target_mask': torch.from_numpy(~np.isnan(target)),
            'location': loc,
            'target_period': torch.from_numpy(s['doy'][tgt].astype(np.int64)),
            'target_doy': torch.from_numpy(s['actual_doy'][tgt].astype(np.int64)),
            'target_year': torch.from_numpy(s['years'][tgt].astype(np.int64)),
            'input_doy': torch.from_numpy(s['actual_doy'][inp].astype(np.int64)),
        }


class DecoupledHistoryDataset(PatchSequenceDataset):
    """
    NIRv and RZSM with independent history lengths.

    Windows are built at max(tlen_nirv, tlen_rzsm) with the usual filtering; within
    each window the shorter channel keeps only its most recent steps, the older ones
    are zeroed and flagged in `history_mask` (T, 2) for the masked encoder.
    """

    def __init__(self, split_dir, tlen_nirv: int, tlen_rzsm: int, **kwargs):
        self.tlen_nirv, self.tlen_rzsm = tlen_nirv, tlen_rzsm
        super().__init__(split_dir, input_bands=(BANDS['NIRv'], BANDS['RZSM']),
                         tlen=max(tlen_nirv, tlen_rzsm), **kwargs)

    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        mask = torch.ones(self.tlen, 2)
        for channel, length in enumerate((self.tlen_nirv, self.tlen_rzsm)):
            pad = self.tlen - length
            sample['inputs'][:pad, channel] = 0.0
            mask[:pad, channel] = 0.0
        sample['history_mask'] = mask
        return sample


class DayOfYearDataset(PatchSequenceDataset):
    """Adds sin/cos day-of-year for the input window and the forecast steps."""

    @staticmethod
    def encode(days: torch.Tensor, height: int, width: int) -> torch.Tensor:
        angle = 2 * np.pi * days.float() / 365.25
        doy = torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)
        return doy[:, :, None, None].expand(-1, -1, height, width).contiguous()

    def __getitem__(self, idx: int):
        sample = super().__getitem__(idx)
        h, w = sample['inputs'].shape[-2:]
        sample['doy'] = self.encode(sample['input_doy'], h, w)
        sample['future_doy'] = self.encode(sample['target_doy'], h, w)
        return sample


def model_kwargs(batch: dict, device) -> dict:
    """Optional model inputs carried by the specialised datasets."""
    return {k: batch[k].to(device) for k in ('doy', 'future_doy', 'history_mask') if k in batch}
