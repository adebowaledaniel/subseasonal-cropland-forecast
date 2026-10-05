"""
Standardised rank cross-correlation and the significance test for autocorrelated series of Lun et al. (2023),
ported from the corTESTsrd R package.
"""

import numpy as np
from scipy import stats


def rank_by_day_of_year(values: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
    """
    Rank each pixel's values among the years sharing the same day of year, scaled to
    (0, 1) as rank / (n + 1). values: (T, ...); NaNs stay NaN.
    """
    ranks = np.full(values.shape, np.nan)
    for day in np.unique(day_of_year):
        idx = day_of_year == day
        group = values[idx]
        n = np.sum(~np.isnan(group), axis=0)
        ranks[idx] = stats.rankdata(group, axis=0, nan_policy='omit') / (n + 1)
    return ranks


def _rank_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    r = stats.rankdata(x)
    r = r - r.mean()
    gamma0 = np.mean(r ** 2)
    if gamma0 == 0:
        return np.zeros(max_lag)
    return np.array([np.sum(r[:-j] * r[j:]) / len(r) for j in range(1, max_lag + 1)]) / gamma0


def spearman_srd_test(x: np.ndarray, y: np.ndarray):
    """
    Spearman correlation of two NaN-free series and its two-sided p-value, with the
    variance inflated by the long-run variance of the rank autocorrelations
    (quartic kernel, bandwidth 3 n^(1/4)).
    """
    n = len(x)
    rho = stats.spearmanr(x, y).statistic
    lags = np.arange(1, n - 1)
    z = lags / (3 * n ** 0.25)
    kernel = np.where(np.abs(z) <= 1, (1 - z ** 2) ** 2, 0.0)
    long_run_var = 1 + 2 * np.sum(kernel * _rank_acf(x, n - 2) * _rank_acf(y, n - 2))
    sd = np.sqrt(long_run_var / n)
    if sd == 0:
        return rho, 1.0 if rho == 0 else 0.0
    return rho, 2 * (1 - stats.norm.cdf(abs(rho), scale=sd))
