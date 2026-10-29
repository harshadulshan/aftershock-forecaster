"""
aftershock_model.py
Real earthquake aftershock forecasting model: the modified Omori-Utsu /
Reasenberg-Jones formulation that the USGS uses operationally to publish
public aftershock probability forecasts on earthquake event pages.

References (used for both the formula and the parameter values below):
  - Reasenberg, P.A. & Jones, L.M. (1989). "Earthquake Hazard After a
    Mainshock in California." Science, 243, 1173-1176.
  - Reasenberg, P.A. & Jones, L.M. (1994). "Earthquake aftershocks: update."
    Science, 265, 1251-1252.
  - Generic California mean parameters a=-1.67, b=0.91, p=1.08, c=0.05
    (Reasenberg & Jones 1989; restated in USGS Open-File Report 96-266,
    which notes that a, b and p are approximately normally distributed
    across California sequences but does not give the spread).
  - Spread of `a` between sequences: Page, M.T. et al. (2016), "Three
    ingredients for improved global aftershock forecasts", Seismological
    Research Letters 87(6), model it as a normal distribution with a
    standard deviation of about 0.5 at best (sigma_0 = 0.49) and about 0.7
    across their global stack. Hardebeck, J.L. et al. (2019), "Updated
    California aftershock parameters", SRL 90(1), 262-270,
    doi:10.1785/0220180240, give the California-specific normal
    distribution the USGS now uses. sigma = 0.5 is used here as a round,
    deliberately modest value from that range: it is the spread that
    matters for a single new sequence.
  - Page, M.T. et al. (2016) and the USGS Operational Aftershock
    Forecasting (OAF) system (earthquake.usgs.gov/data/oaf/background.php)
    describe the same rate model plus a Bayesian treatment of parameter
    uncertainty in `a`, which this module reproduces via a discrete
    grid-approximation posterior (simpler to implement correctly than the
    production system's full Bayesian machinery, but the same idea:
    combine a generic prior with sequence-specific observed counts).

Model
-----
Aftershock occurrence rate (events per day, magnitude >= M) at time t
(days) after a mainshock of magnitude Mmain:

    lambda(t) = 10^(a + b*(Mmain - M)) / (t + c)^p

Expected number of events in a window [t1, t2] is the integral of
lambda(t) dt, which has a closed form for p != 1:

    N(t1,t2) = 10^(a + b*(Mmain-M)) / (1-p) * [(t2+c)^(1-p) - (t1+c)^(1-p)]

and for p == 1:

    N(t1,t2) = 10^(a + b*(Mmain-M)) * [ln(t2+c) - ln(t1+c)]

Given N, the number of aftershocks in the window is modeled as Poisson, so
the probability of at least one event is:

    P(N>=1) = 1 - exp(-N)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# numpy 2.0 renamed trapz -> trapezoid; support both so this runs on either
# version without the caller needing to know which numpy is installed.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# Generic California parameters (Reasenberg & Jones, 1989). The prior sigma
# on `a` is the between-sequence spread (~0.5, Page et al. 2016), not
# the much smaller standard error of the generic mean.
GENERIC_A = -1.67
GENERIC_A_SIGMA = 0.5
GENERIC_B = 0.91
GENERIC_C = 0.05
GENERIC_P = 1.08


def rate(t_days: np.ndarray, mainshock_mag: float, mag_threshold: float,
         a: float = GENERIC_A, b: float = GENERIC_B, c: float = GENERIC_C, p: float = GENERIC_P) -> np.ndarray:
    """Instantaneous aftershock rate (events/day, M >= mag_threshold) at t_days after the mainshock."""
    productivity = 10 ** (a + b * (mainshock_mag - mag_threshold))
    return productivity / np.power(np.asarray(t_days, dtype=float) + c, p)


def expected_count(t1: float, t2: float, mainshock_mag: float, mag_threshold: float,
                    a: float = GENERIC_A, b: float = GENERIC_B, c: float = GENERIC_C, p: float = GENERIC_P) -> float:
    """Expected number of M >= mag_threshold aftershocks in the window [t1, t2] days after the mainshock."""
    if t2 <= t1:
        return 0.0
    productivity = 10 ** (a + b * (mainshock_mag - mag_threshold))
    if abs(p - 1.0) < 1e-9:
        return float(productivity * (np.log(t2 + c) - np.log(t1 + c)))
    return float(productivity / (1 - p) * ((t2 + c) ** (1 - p) - (t1 + c) ** (1 - p)))


def completeness_magnitude(magnitudes, floor: float, min_events: int = 20,
                           bin_width: float = 0.1, correction: float = 0.2) -> float:
    """Estimate the magnitude of completeness Mc: the smallest magnitude the
    catalog records reliably for this sequence.

    Below Mc many events are missed, so counting them as if complete makes a
    sequence look far less productive than it is. Outside dense networks (the
    US, Japan, Europe) the global catalog is usually complete only from about
    M4-4.5, and right after a big mainshock even less.

    With at least `min_events` events this is the maximum-curvature estimate
    (the most common 0.1-magnitude bin) plus the usual +0.2 correction
    (Woessner & Wiemer, 2005). With fewer events it falls back to the smallest
    magnitude observed. The result is never below `floor`, the catalog's
    search threshold.
    """
    mags = np.asarray([m for m in magnitudes if m is not None and np.isfinite(m)], dtype=float)
    if len(mags) == 0:
        return float(floor)
    if len(mags) >= min_events:
        bins = np.round(np.round(mags / bin_width) * bin_width, 1)
        values, counts = np.unique(bins, return_counts=True)
        mc = float(values[np.argmax(counts)]) + correction
    else:
        mc = float(mags.min())
    return float(round(max(mc, floor), 1))


def prob_at_least_one(expected_n: float) -> float:
    return float(1 - np.exp(-expected_n))


@dataclass
class ForecastRow:
    window_label: str
    mag_threshold: float
    expected_n_mean: float
    expected_n_low: float
    expected_n_high: float
    prob_mean: float
    prob_low: float
    prob_high: float


def posterior_over_a(
    observed_count: int,
    obs_t1: float,
    obs_t2: float,
    mainshock_mag: float,
    obs_mag_threshold: float,
    prior_mean: float = GENERIC_A,
    prior_sigma: float = GENERIC_A_SIGMA,
    b: float = GENERIC_B, c: float = GENERIC_C, p: float = GENERIC_P,
    grid_width_sigmas: float = 6.0, grid_points: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """Bayesian update of the productivity parameter `a` given an observed
    aftershock count in some early window of the actual sequence, combining
    the generic-California prior with a Poisson likelihood from the
    sequence's own data — the same idea used by the USGS operational
    aftershock forecasting system to adapt the generic model to a specific
    earthquake as its aftershocks unfold.

    Returns (a_grid, posterior_density) suitable for Monte-Carlo sampling
    or direct numerical summarization.
    """
    a_grid = np.linspace(prior_mean - grid_width_sigmas * prior_sigma,
                          prior_mean + grid_width_sigmas * prior_sigma, grid_points)
    prior = np.exp(-0.5 * ((a_grid - prior_mean) / prior_sigma) ** 2)

    # Poisson likelihood of observing `observed_count` given each candidate a
    lambdas = np.array([
        expected_count(obs_t1, obs_t2, mainshock_mag, obs_mag_threshold, a=a_val, b=b, c=c, p=p)
        for a_val in a_grid
    ])
    lambdas = np.clip(lambdas, 1e-12, None)
    # log Poisson pmf (drop the constant factorial term — irrelevant after normalization)
    log_lik = observed_count * np.log(lambdas) - lambdas
    log_lik -= log_lik.max()
    likelihood = np.exp(log_lik)

    posterior = prior * likelihood
    total = _trapz(posterior, a_grid)
    if total <= 0 or not np.isfinite(total):
        # Degenerate case (e.g. zero-variance grid) — fall back to prior
        posterior = prior
        total = _trapz(posterior, a_grid)
    posterior = posterior / total
    return a_grid, posterior


def forecast(
    mainshock_mag: float,
    now_t_days: float,
    windows: dict[str, tuple[float, float]],
    mag_thresholds: list[float],
    a_grid: Optional[np.ndarray] = None,
    a_posterior: Optional[np.ndarray] = None,
    b: float = GENERIC_B, c: float = GENERIC_C, p: float = GENERIC_P,
    n_samples: int = 4000,
    random_state: int = 0,
) -> list[ForecastRow]:
    """Monte Carlo forecast: sample `a` from its (posterior or generic prior)
    distribution, compute expected count and probability for each
    (window, magnitude threshold) pair, and summarize as mean + 90% interval.
    """
    rng = np.random.default_rng(random_state)
    if a_grid is None or a_posterior is None:
        a_samples = rng.normal(GENERIC_A, GENERIC_A_SIGMA, n_samples)
    else:
        probs = a_posterior / a_posterior.sum()
        a_samples = rng.choice(a_grid, size=n_samples, p=probs)

    rows = []
    for label, (t1, t2) in windows.items():
        for M in mag_thresholds:
            n_vals = np.array([
                expected_count(t1, t2, mainshock_mag, M, a=a_val, b=b, c=c, p=p)
                for a_val in a_samples
            ])
            probs = 1 - np.exp(-n_vals)
            rows.append(ForecastRow(
                window_label=label,
                mag_threshold=M,
                expected_n_mean=float(n_vals.mean()),
                expected_n_low=float(np.percentile(n_vals, 5)),
                expected_n_high=float(np.percentile(n_vals, 95)),
                prob_mean=float(probs.mean()),
                prob_low=float(np.percentile(probs, 5)),
                prob_high=float(np.percentile(probs, 95)),
            ))
    return rows
