"""Forecast skill metrics."""

from typing import Dict, Optional

import numpy as np

WATER_LIMITED, ENERGY_LIMITED = 1, 2


def acc(pred: np.ndarray, obs: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Anomaly correlation coefficient against a zero climatology.

    The inputs are already day-of-year anomalies, so ACC = sum(p o) / sqrt(sum(p^2) sum(o^2)).
    Values outside `mask` (broadcast against the arrays) and NaNs are excluded.
    De-standardising first gives the same value: the pooled training mean is ~0 and
    ACC does not depend on the scale.
    """
    p = np.asarray(pred, dtype=np.float64)
    o = np.asarray(obs, dtype=np.float64)
    keep = ~(np.isnan(p) | np.isnan(o))
    if mask is not None:
        keep &= np.broadcast_to(mask, p.shape)
    p, o = np.where(keep, p, 0.0), np.where(keep, o, 0.0)
    denom = np.sqrt(np.sum(p ** 2) * np.sum(o ** 2))
    return float(np.sum(p * o) / denom) if denom > 0 else float('nan')


def regression_metrics(pred: np.ndarray, obs: np.ndarray,
                       mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    p = np.asarray(pred, dtype=np.float64)
    o = np.asarray(obs, dtype=np.float64)
    if mask is not None:
        p, o = p[mask], o[mask]
    err = p - o
    ss_tot = np.sum((o - o.mean()) ** 2)
    return {
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'mae': float(np.mean(np.abs(err))),
        'bias': float(np.mean(err)),
        'r2': float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float('nan'),
    }


def rmse(pred: np.ndarray, obs: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    return regression_metrics(pred, obs, mask)['rmse']


def forecast_skill(pred: np.ndarray, obs: np.ndarray, mask: np.ndarray,
                   aridity: Optional[np.ndarray] = None, mask_regression: bool = False) -> Dict:
    """
    Test-set skill in the form reported in the paper.

    pred, obs, mask: (N, horizon, H, W); unobserved targets are zero in `obs`.
    aridity: optional (N, H, W) class of each sample's pixels (1 water-, 2 energy-limited).

    RMSE, MAE, R2 and bias are taken over every pixel of the forecast patches, or over
    observed pixels only with mask_regression=True (the persistence, ridge and XGBoost
    baselines). The climatology (zero-anomaly) RMSE, the skill-score denominator and the
    per-lead RMSE use observed pixels only. ACC is computed on the zero-filled observations.
    """
    horizon = pred.shape[1]
    out = regression_metrics(pred, obs, mask if mask_regression else None)
    out['clim_rmse'] = rmse(np.zeros_like(obs), obs, mask)
    out['skill_score'] = 1 - out['rmse'] / out['clim_rmse']
    out['acc'] = acc(pred, obs)
    out['acc_lead'] = [acc(pred[:, t], obs[:, t]) for t in range(horizon)]
    out['rmse_lead'] = [rmse(pred[:, t], obs[:, t], mask[:, t]) for t in range(horizon)]
    out['clim_rmse_lead'] = [rmse(np.zeros_like(obs[:, t]), obs[:, t], mask[:, t])
                             for t in range(horizon)]
    if aridity is not None:
        for name, value in (('water_limited', WATER_LIMITED), ('energy_limited', ENERGY_LIMITED)):
            regime = aridity == value
            out[f'{name}_acc_lead'] = [acc(pred[:, t], obs[:, t], regime) for t in range(horizon)]
    return out


def pixelwise_acc(pred: np.ndarray, obs: np.ndarray, min_samples: int = 5) -> np.ndarray:
    """
    Per-pixel ACC: Pearson correlation over forecasts, (N, H, W) -> (H, W).
    Pixels with fewer than `min_samples` valid pairs are NaN.
    """
    valid = ~(np.isnan(pred) | np.isnan(obs))
    n = valid.sum(axis=0)
    p = np.where(valid, pred, 0.0)
    o = np.where(valid, obs, 0.0)
    p = np.where(valid, p - p.sum(0) / np.maximum(n, 1), 0.0)
    o = np.where(valid, o - o.sum(0) / np.maximum(n, 1), 0.0)
    denom = np.sqrt((p ** 2).sum(0) * (o ** 2).sum(0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where((denom > 0) & (n >= min_samples), (p * o).sum(0) / denom, np.nan)
