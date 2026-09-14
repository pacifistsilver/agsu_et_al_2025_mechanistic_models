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



