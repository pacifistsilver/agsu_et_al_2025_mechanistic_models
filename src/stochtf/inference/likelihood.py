"""Exact log-likelihood of observed counts under the promoter models.

This replaces the ABC layer. Previously inference compared a Gillespie
simulation against the data through ``summary_stat``, which reduced both to
``[mean, fano]`` -- two numbers -- and accepted parameters within an epsilon
ball of that. Everything else about the distribution was discarded, and the
simulator itself was noisy, so the ABC posterior mixed genuine parameter
uncertainty with Monte Carlo error.

The stationary distribution is available exactly (see
:mod:`stochtf.analytical.pgf`), so the likelihood of iid counts is just

    log L(theta) = sum_i log P(y_i | theta)

evaluated on the whole distribution, deterministically. There is no tolerance
to tune and no simulation noise, and the sampler targets the true posterior
rather than an ABC approximation to it.
"""

import numpy as np

from stochtf.analytical import pgf

#: Counts below this are treated as numerically zero probability. The floor
#: keeps a single unlucky observation from sending the whole log-likelihood to
#: -inf and stalling the sampler.
MIN_PROB = 1e-300


def prepare_counts(counts):
    """Validate observed counts and return them as a non-negative integer array."""
    y = np.asarray(counts)
    if y.ndim > 1:
        y = y.ravel()
    if not np.all(np.isfinite(y)):
        raise ValueError("counts contain non-finite values")
    y_int = np.rint(y).astype(np.int64)
    if np.any(np.abs(y - y_int) > 1e-8):
        raise ValueError("counts must be integers (molecule numbers)")
    if np.any(y_int < 0):
        raise ValueError("counts must be non-negative")
    return y_int


def log_likelihood(counts, a_s, b_s, a_n, b_n, k_y, gamma, model="dimer",
                   y_max=None):
    """Exact log-likelihood of iid stationary counts.

    ``counts`` are molecule numbers, one per cell. ``model`` selects the
    promoter logic via :data:`stochtf.analytical.pgf.MODEL_GATE`.

    Returns ``-inf`` for parameters that are out of the model's support rather
    than raising, so a sampler can simply reject them.
    """
    y = prepare_counts(counts)

    if not all(np.isfinite([a_s, b_s, a_n, b_n, k_y, gamma])):
        return -np.inf
    if min(a_s, b_s, a_n, b_n, k_y) <= 0 or gamma <= 0:
        return -np.inf

    # The grid has to reach the largest observation, whatever the mean implies.
    if y_max is None:
        gate = pgf.MODEL_GATE[model]
        mean, second = pgf.factorial_moments(a_s, b_s, a_n, b_n, k_y, gamma,
                                             gate, order=2)
        var = second + mean - mean**2
        y_max = int(mean + 12.0 * np.sqrt(max(var, 1.0)) + 12)
    y_max = max(int(y.max()), y_max, 32)

    try:
        p = pgf.model_pmf(model, a_s, b_s, a_n, b_n, k_y, gamma, y_max=y_max)
    except (FloatingPointError, np.linalg.LinAlgError):
        return -np.inf

    if not np.all(np.isfinite(p)):
        return -np.inf

    return float(np.sum(np.log(np.maximum(p[y], MIN_PROB))))


def log_likelihood_factory(counts, model="dimer", k_y=None, gamma=None):
    """Build ``f(a_s, a_n, b_s, b_n, ...) -> float`` bound to fixed data.

    The argument order matches the priors declared in
    :mod:`stochtf.inference.models` (alpha_s, alpha_n, beta_s, beta_n), which is
    *not* the order :func:`log_likelihood` takes -- the mismatch is handled here
    once rather than at every call site.

    ``k_y`` and ``gamma`` may be pinned here, in which case the returned
    function takes four arguments; leave them None to infer them too, and the
    returned function takes six.
    """
    y = prepare_counts(counts)
    fixed = k_y is not None and gamma is not None

    if fixed:
        def f(a_s, a_n, b_s, b_n):
            return log_likelihood(y, a_s, b_s, a_n, b_n, k_y, gamma, model)
    else:
        def f(a_s, a_n, b_s, b_n, k_y_, gamma_):
            return log_likelihood(y, a_s, b_s, a_n, b_n, k_y_, gamma_, model)

    f.n_params = 4 if fixed else 6
    return f
