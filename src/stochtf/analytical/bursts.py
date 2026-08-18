"""General burst frequency / size for ANY autonomous driver chain.

Definitions that work even with graded output or a fluctuating pool:
  OFF set F = driver states with zero emission;  ON set O = the rest.
  f = stationary flux across F -> O   (every burst starts with exactly one such jump)
  burst size = molecules emitted between entering O and returning to F
             = discrete phase-type on O, allowing per-state emission rates
  b = mean burst size;  identity  <y> = b f / gamma  must hold.
"""
import numpy as np
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import spsolve

def stationary(Q):
    n = Q.shape[0]
    M = csc_matrix(Q).T.tolil(); M[0, :] = 1.0
    r = np.zeros(n); r[0] = 1.0
    return spsolve(csc_matrix(M), r)

def burst_stats(Q, kvec, gamma, mmax=4000):
    """Returns f, b, pmf, <y>."""
    Q = np.asarray(csc_matrix(Q).todense())
    kvec = np.asarray(kvec, float)
    pi = stationary(csc_matrix(Q))
    ON = np.where(kvec > 0)[0]; OFF = np.where(kvec <= 0)[0]

    # burst frequency = stationary flux OFF -> ON
    f = sum(pi[i] * Q[i, j] for i in OFF for j in ON)

    # entry distribution into ON, weighted by that flux
    ent = np.array([sum(pi[i] * Q[i, j] for i in OFF) for j in ON]); ent /= ent.sum()

    R = np.array([-Q[j, j] for j in ON])                  # total exit rate
    kO = kvec[ON]
    tot = kO + R
    B = np.zeros((len(ON), len(ON)))
    for a, j in enumerate(ON):
        for c, jj in enumerate(ON):
            if a != c: B[a, c] = Q[j, jj] / tot[a]
    t = np.array([sum(Q[j, i] for i in OFF) / tot[a] for a, j in enumerate(ON)])
    A = np.diag(kO / tot)

    S = np.linalg.inv(np.eye(len(ON)) - B)
    U, v = S @ A, S @ t
    pmf = np.empty(mmax + 1); row = ent.copy()
    for m in range(mmax + 1):
        pmf[m] = row @ v; row = row @ U
    m = np.arange(pmf.size)
    b = (m * pmf).sum()
    return f, b, pmf, (kvec * pi).sum() / gamma

def rate_burst_stats(topology, a_s, a_n, k_y, b_s, b_n):
    """(ON fraction, burst frequency, mean burst size) in closed form.

    The same f and b as ``burst_stats``, for the two promoters the inference
    layer fits, where the algebra closes: the OFF set is one silent state, so a
    burst is a single excursion out of it, and every ON state emits at k_y.

        tau_off = 1 / (a_s + a_n)   mean wait in the silent state
        tau_on                      mean length of one excursion -- the only
                                    part that differs between the topologies
        f = 1 / (tau_on + tau_off)  cycles per unit time, i.e. the OFF -> ON flux
        b = k_y tau_on              molecules made per excursion

    so <y> = b f / gamma holds exactly here, not only in the bursty limit.
    Rates are in units of gamma. The off-rates are arguments rather than
    defaults because they are pinned by the inference layer, which sits above
    this module.
    """
    if a_s + a_n <= 0:
        return np.nan, np.nan, np.nan
    tau_off = 1.0 / (a_s + a_n)
    if topology == "heterodimer":
        # Silent only when neither site is bound; the sites are independent.
        silent = (b_s / (a_s + b_s)) * (b_n / (a_n + b_n))
        tau_on = tau_off * (1.0 - silent) / silent if silent > 0 else np.inf
    else:
        # Exclusive occupancy: the excursion is S or N, and only one of them.
        entry_s = a_s / (a_s + a_n)
        tau_on = entry_s / b_s + (1.0 - entry_s) / b_n
    on_fraction = tau_on / (tau_on + tau_off)
    return on_fraction, 1.0 / (tau_on + tau_off), k_y * tau_on
