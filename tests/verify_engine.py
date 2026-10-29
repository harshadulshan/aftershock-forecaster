"""
Verification suite for aftershock_model.py.

Offline: no USGS calls. The closed-form integrals are checked against
numerical integration, the Bayesian update against an independent
brute-force posterior, and parameter recovery on simulated sequences.

    python tests/verify_engine.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy import integrate, stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aftershock_model import (  # noqa: E402
    GENERIC_A, GENERIC_A_SIGMA, GENERIC_B, GENERIC_C, GENERIC_P,
    completeness_magnitude, expected_count, forecast, posterior_over_a, prob_at_least_one, rate,
)

PASSED, FAILED = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------------------
print("\n1. Reasenberg-Jones rate and expected counts")
check("rate = 10^(a + b(Mm - M)) / (t + c)^p",
      np.isclose(rate(np.array([2.0]), 7.0, 5.0)[0],
                 10 ** (GENERIC_A + GENERIC_B * 2.0) / (2.0 + GENERIC_C) ** GENERIC_P))
for (t1, t2) in [(0, 1), (0, 7), (1, 30), (7, 365)]:
    num, _ = integrate.quad(lambda t: rate(np.array([t]), 6.5, 3.0)[0], t1, t2, limit=200)
    check(f"closed-form count matches numerical integral on [{t1}, {t2}] days",
          np.isclose(expected_count(t1, t2, 6.5, 3.0), num, rtol=1e-7), f"{num:.4f}")
num, _ = integrate.quad(lambda t: rate(np.array([t]), 6.5, 3.0, p=1.0)[0], 0, 30)
check("p = 1 uses the logarithmic form", np.isclose(expected_count(0, 30, 6.5, 3.0, p=1.0), num, rtol=1e-7))
check("windows add up: N(0,30) = N(0,7) + N(7,30)",
      np.isclose(expected_count(0, 30, 7, 4), expected_count(0, 7, 7, 4) + expected_count(7, 30, 7, 4)))
check("an empty or reversed window has 0 expected events", expected_count(5, 5, 7, 4) == 0 and expected_count(5, 2, 7, 4) == 0)
check("lowering the threshold by 1 magnitude multiplies counts by 10^b",
      np.isclose(expected_count(0, 7, 7, 3) / expected_count(0, 7, 7, 4), 10 ** GENERIC_B))
check("a bigger mainshock is more productive", expected_count(0, 7, 7.5, 4) > expected_count(0, 7, 6.5, 4))
check("Omori decay: the second week expects fewer events than the first",
      expected_count(7, 14, 7, 4) < expected_count(0, 7, 7, 4))
check("P(>=1) = 1 - exp(-N)", np.isclose(prob_at_least_one(2.0), 1 - np.exp(-2.0)) and prob_at_least_one(0.0) == 0.0)

# Worked example: generic California M7, chance of an M>=5 in the first week.
n = expected_count(0, 7, 7.0, 5.0)
hand = 10 ** (-1.67 + 0.91 * 2) / (1 - 1.08) * ((7.05) ** -0.08 - 0.05 ** -0.08)
check("worked example: M7, M>=5 aftershocks in the first 7 days", np.isclose(n, hand), f"N = {n:.3f}, P = {prob_at_least_one(n):.1%}")

# ---------------------------------------------------------------------------
print("\n2. Bayesian update of productivity a")
args = dict(obs_t1=0.0, obs_t2=3.0, mainshock_mag=6.8, obs_mag_threshold=3.0)
for k_obs in (0, 5, 40):
    grid, post = posterior_over_a(k_obs, **args)
    lam = np.array([expected_count(0, 3, 6.8, 3.0, a=a) for a in grid])
    brute = stats.norm.pdf(grid, GENERIC_A, GENERIC_A_SIGMA) * stats.poisson.pmf(k_obs, lam)
    brute /= integrate.trapezoid(brute, grid)
    check(f"posterior matches prior x Poisson likelihood (k = {k_obs})", np.allclose(post, brute, atol=1e-6 * brute.max()))
    check(f"posterior integrates to 1 (k = {k_obs})", np.isclose(integrate.trapezoid(post, grid), 1.0))

prior_n = expected_count(0, 3, 6.8, 3.0)
grid, post0 = posterior_over_a(0, **args)
grid, post_hi = posterior_over_a(int(10 * prior_n), **args)
mean = lambda p: integrate.trapezoid(grid * p, grid)  # noqa: E731
check("fewer aftershocks than expected pulls a below the prior mean", mean(post0) < GENERIC_A, f"{mean(post0):.2f}")
check("10x more aftershocks than expected pushes a above it", mean(post_hi) > GENERIC_A, f"{mean(post_hi):.2f}")

# Recovery: simulate many sequences with a known true a and check the posterior finds it.
rng = np.random.default_rng(0)
true_a = -1.0
errors, covered = [], 0
for _ in range(200):
    lam_true = expected_count(0, 7, 6.5, 2.5, a=true_a)
    k = rng.poisson(lam_true)
    g, p = posterior_over_a(k, 0, 7, 6.5, 2.5)
    m = integrate.trapezoid(g * p, g)
    cdf = integrate.cumulative_trapezoid(p, g, initial=0)
    lo, hi = np.interp([0.05, 0.95], cdf, g)
    errors.append(m - true_a)
    covered += lo <= true_a <= hi
bias = float(np.mean(errors))
check("recovers a true a = -1.0 from simulated 7-day sequences", abs(bias) < 0.05, f"mean error {bias:+.3f}")
check("90% posterior interval covers the true a at least ~90% of the time", 0.85 <= covered / 200 <= 0.97, f"{covered / 2:.1f}%")

# ---------------------------------------------------------------------------
print("\n3. Monte Carlo forecast")
windows = {"Next day": (3, 4), "Next week": (3, 10), "Next month": (3, 33)}
rows = forecast(7.0, 3.0, windows, [4.0, 5.0, 6.0], n_samples=20000)
by = {(r.window_label, r.mag_threshold): r for r in rows}
check("one row per window x magnitude", len(rows) == 9)
check("interval brackets the mean", all(r.prob_low <= r.prob_mean <= r.prob_high and r.expected_n_low <= r.expected_n_mean <= r.expected_n_high for r in rows))
check("probability falls as the magnitude threshold rises",
      all(by[(w, 4.0)].prob_mean > by[(w, 5.0)].prob_mean > by[(w, 6.0)].prob_mean for w in windows))
check("probability rises with a longer window",
      all(by[("Next day", m)].prob_mean < by[("Next week", m)].prob_mean < by[("Next month", m)].prob_mean for m in (4.0, 5.0, 6.0)))
# Under the prior, a ~ Normal(mu, sigma) so 10^a is lognormal: E[N] = N(mu) * exp((sigma ln 10)^2 / 2).
base = expected_count(3, 10, 7.0, 5.0)
lognormal_mean = base * np.exp((GENERIC_A_SIGMA * np.log(10)) ** 2 / 2)
check("prior-only mean count matches the lognormal mean",
      np.isclose(by[("Next week", 5.0)].expected_n_mean, lognormal_mean, rtol=0.05),
      f"{by[('Next week', 5.0)].expected_n_mean:.3f} vs {lognormal_mean:.3f}")
q_lo = base * 10 ** (GENERIC_A_SIGMA * stats.norm.ppf(0.05))
check("prior-only 5th percentile matches 10^(mu + z_0.05 sigma)",
      np.isclose(by[("Next week", 5.0)].expected_n_low, q_lo, rtol=0.05))
narrow = posterior_over_a(500, 0, 3, 7.0, 3.0)
rows_post = forecast(7.0, 3.0, windows, [5.0], a_grid=narrow[0], a_posterior=narrow[1], n_samples=20000)
check("with lots of data the forecast interval gets much narrower",
      (rows_post[1].expected_n_high / rows_post[1].expected_n_low) < 0.25 * (by[("Next week", 5.0)].expected_n_high / by[("Next week", 5.0)].expected_n_low))
check("same seed, same forecast (reproducible)",
      forecast(7.0, 3.0, windows, [5.0])[0] == forecast(7.0, 3.0, windows, [5.0])[0])

# ---------------------------------------------------------------------------
print("\n4. Catalog completeness")
rng = np.random.default_rng(1)
true_mc = 4.3
mags = true_mc - 0.05 + rng.exponential(1 / (GENERIC_B * np.log(10)), 3000)  # Gutenberg-Richter, reported to 0.1
below = 2.5 + rng.uniform(0, true_mc - 2.5, 200)                          # a few detected small events
cat = np.round(np.r_[mags, below], 1)
mc_est = completeness_magnitude(cat, floor=2.5)
check("maximum curvature + 0.2 gives Mc = 4.5 for a catalog complete from 4.3", np.isclose(mc_est, 4.5), f"Mc = {mc_est}")
check("small samples fall back to the smallest magnitude found", completeness_magnitude([4.8, 4.2, 4.8, 5.3], floor=2.5) == 4.2)
check("never below the search threshold", completeness_magnitude([2.0, 2.1, 3.0], floor=2.5) == 2.5)
check("no events: the search threshold", completeness_magnitude([], floor=2.5) == 2.5)

# A complete sequence above Mc: counting it from Mc recovers the true a; counting from
# 2.5 (where the catalog is incomplete) badly underestimates it.
true_a = -1.8
lam = expected_count(0, 5, 6.5, 4.3, a=true_a)
k = int(round(lam))
g_ok, p_ok = posterior_over_a(k, 0, 5, 6.5, 4.3)
g_bad, p_bad = posterior_over_a(k, 0, 5, 6.5, 2.5)
m_ok = integrate.trapezoid(g_ok * p_ok, g_ok)
m_bad = integrate.trapezoid(g_bad * p_bad, g_bad)
check("counting from Mc recovers productivity", abs(m_ok - true_a) < 0.2, f"a = {m_ok:.2f} vs true {true_a}")
check("counting an incomplete catalog from M2.5 underestimates it by over 1 unit", m_ok - m_bad > 1.0, f"a = {m_bad:.2f}")

# ---------------------------------------------------------------------------
print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    print("Failed:", *FAILED, sep="\n  - ")
    sys.exit(1)
