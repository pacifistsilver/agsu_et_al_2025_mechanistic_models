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
