"""Which parameters stationary counts can actually determine.

Fitting a model that the data cannot constrain produces posteriors that look
like answers but are really priors. This module measures that directly, before
any sampling, from the Fisher information of the exact stationary distribution.

For iid counts the information about theta is

    I_jk = n * sum_y  (dP(y)/dtheta_j)(dP(y)/dtheta_k) / P(y),

computed here in *log* parameter space, so the resulting standard errors read as
fractional errors: 0.1 means "determined to about 10%".

What this found for the two-site promoter
-----------------------------------------
A single-site telegraph promoter (alpha, beta, k_y) is well identified -- around
7%, 10% and 1% at n = 800, matching the published Poisson-Beta fits to
single-cell data. Switching on the second site, in the *same* bursty regime,
sends the four switching rates to standard errors of 600-5000%: only k_y and one
occupancy-like combination survive. Of five log-parameter directions, two are
determined, one is weak and two are flat.

That is a property of the model and the data type, not of the sampler or the
priors, and no choice of parameter regime repairs it. Stationary counts are a
one-dimensional summary of a four-state promoter; there is not enough left in
P(y) to separate four switching rates.

There is also an exact discrete degeneracy: the likelihood is invariant under
exchanging the two sites, (alpha_s, beta_s) <-> (alpha_n, beta_n), for every
gate. So the sites are identifiable at best as an unordered pair, and marginal
posterior means for alpha_s and alpha_n average over both labellings unless the
sampler is given an ordering constraint. See :func:`exchange_symmetry_residual`.
"""

import numpy as np

from stochtf.analytical import pgf

PARAM_NAMES = ("alpha_s", "beta_s", "alpha_n", "beta_n", "k_y")

#: Fractional standard error below which a parameter is called determined.
DETERMINED_SE = 0.5


def _pmf_grid(theta, gate, y_max):
    a_s, b_s, a_n, b_n, k_y = theta
    return pgf.stationary_pmf(a_s, b_s, a_n, b_n, k_y, 1.0, gate, y_max=y_max)


def _default_y_max(theta, gate):
    a_s, b_s, a_n, b_n, k_y = theta
    mean, var, _ = pgf.moments(a_s, b_s, a_n, b_n, k_y, 1.0, gate)
    return int(mean + 14.0 * np.sqrt(max(var, 1.0)) + 30)


def fisher_information(theta, n_obs, gate="OR", free=None, h=1e-5, y_max=None):
    """Fisher information in log-parameter space.

    ``theta`` is (alpha_s, beta_s, alpha_n, beta_n, k_y) with gamma = 1.
    ``free`` selects a subset of indices to treat as unknown; the rest are held
    fixed, which is how the single-site case is obtained (freeze site n).

    Uses first derivatives only. A finite-difference Hessian of the expected
    log-likelihood is the textbook route but loses far too much precision here --
    it returned negative eigenvalues on distributions this flat.
    """
    theta = np.asarray(theta, dtype=float)
    if free is None:
        free = range(len(theta))
    free = list(free)
    if y_max is None:
        y_max = _default_y_max(theta, gate)

    log_theta = np.log(theta)
    p0 = np.clip(_pmf_grid(theta, gate, y_max), 1e-300, None)

    derivs = np.empty((len(free), p0.size))
    for row, i in enumerate(free):
        step = np.zeros_like(log_theta)
        step[i] = h
        plus = _pmf_grid(np.exp(log_theta + step), gate, y_max)
        minus = _pmf_grid(np.exp(log_theta - step), gate, y_max)
        derivs[row] = (plus - minus) / (2 * h)

    return n_obs * (derivs / p0) @ derivs.T


def standard_errors(theta, n_obs, gate="OR", free=None, **kwargs):
    """Fractional standard errors from the Cramer-Rao bound.

    A pseudo-inverse is used because the information matrix is genuinely
    singular in the flat directions; the returned value for those is a lower
    bound on an already enormous error, so treat anything above ~1 as "not
    determined" rather than as a number.
    """
    info = fisher_information(theta, n_obs, gate, free, **kwargs)
    return np.sqrt(np.diag(np.linalg.pinv(info)))


def identifiable_directions(theta, n_obs, gate="OR", **kwargs):
    """Eigen-decomposition of the information matrix.

    Returns ``(eigenvalues, eigenvectors)`` sorted from best- to worst-determined.
    Each eigenvector is a combination of log-parameters; large eigenvalues are
    the combinations the data pins down, near-zero ones are the flat directions
    along which the likelihood is indifferent.
    """
    info = fisher_information(theta, n_obs, gate, **kwargs)
    vals, vecs = np.linalg.eigh(info)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def describe(theta, n_obs, gate="OR", free=None, names=PARAM_NAMES):
    """Human-readable identifiability report for one parameter point."""
    free = list(range(len(theta))) if free is None else list(free)
    se = standard_errors(theta, n_obs, gate, free)
    mean, var, fano = pgf.moments(*theta[:4], theta[4], 1.0, gate)

    lines = [f"gate={gate}  n={n_obs}  mean={mean:.2f}  Fano={fano:.2f}",
             "fractional standard errors:"]
    for idx, value in zip(free, se):
        verdict = "determined" if value < DETERMINED_SE else "NOT identified"
        lines.append(f"    {names[idx]:8s} {value * 100:10.1f}%   {verdict}")
    return "\n".join(lines)


def exchange_symmetry_residual(theta, gate="OR", y_max=400):
    """Max |P(theta) - P(theta with the two sites swapped)|.

    This is zero to rounding for every gate: the two sites are exchangeable, so
    the posterior has two exactly equivalent modes. Any inference that reports
    alpha_s and alpha_n separately needs an ordering constraint to break it.
    """
    a_s, b_s, a_n, b_n, k_y = theta
    p1 = pgf.stationary_pmf(a_s, b_s, a_n, b_n, k_y, 1.0, gate, y_max=y_max)
    p2 = pgf.stationary_pmf(a_n, b_n, a_s, b_s, k_y, 1.0, gate, y_max=y_max)
    return float(np.abs(p1 - p2).max())
