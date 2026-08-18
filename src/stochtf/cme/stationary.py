"""Stationary distributions by sFSP (Gupta, Mikelson & Khammash, 2017).

The FSP implementation in :mod:`stochtf.cme.fsp` integrates p(t) forward from an
initial state. That answers a different question from the one inference asks:
the counts are snapshots of cells at steady state, so what is wanted is the
*stationary* distribution, and integrating to it from t = 0 is slow and
indirect.

Truncating for a stationary solve is not as simple as truncating for a
transient one. Classical FSP sends every transition that leaves the truncation
into an absorbing state, and the only stationary distribution of *that* chain
puts all its mass on the absorbing state -- useless. The common workaround is
to zero the escaping propensities instead ("reflecting" truncation), which
usually works but comes with no guarantee.

sFSP keeps the escaping transitions and *redirects* them to a designated state
inside the truncation, so the projected chain stays irreducible and has a
genuine stationary distribution. Writing Q_n for the sub-generator on the
truncated set E_n (diagonal still carrying the escaping rates) and c_n for the
per-state outflow, the projected generator is

    Qbar_n = Q_n + c_n b_l,                                            (3.14)

with b_l the row vector selecting the designated state x_l -- that is, c_n is
added to column l. Its rows sum to zero, so it is a valid generator.

What that buys is an error certificate. Solving pi_n^T Qbar_n = 0 and taking

    r_out = c_n^T pi_n,          gamma = r_out * ||E_n||_V             (3.15, 3.22)
    ||E_n||_V = V(x_l) + max_{x in B(E_n)} V(x)                        (3.23)

for a Foster-Lyapunov function V, Theorem 3.1 gives two-sided control,

    M' gamma  <=  ||pi - pi_n||_V  <=  M gamma,

so gamma is computable from the output and bounds the error you cannot compute.
Algorithm 1 is then just: solve, evaluate gamma, expand the truncation, repeat.

Reference: A. Gupta, J. Mikelson, M. Khammash, "A finite state projection
algorithm for the stationary solution of the chemical master equation",
J. Chem. Phys. 147, 154101 (2017); arXiv:1704.07259. Equation numbers above are
that paper's.

Specialisation here
-------------------
The chain is (promoter state, mRNA count): the promoter is a bounded species
with a handful of states, mRNA is the single free species, so the state space
has the E_b x N_0 form the paper assumes and the truncation is
E_n = E_b x {0..y_max}. Only transcription escapes it (degradation at y = 0 has
zero propensity), so c_n is supported on the top row of the grid.

The generator is block-tridiagonal in y with blocks the size of the promoter,
which the solve exploits: sweeping in y costs O(y_max) small solves instead of
one large sparse factorisation, and that is what makes this fast enough to sit
inside a sampler.
"""

import numpy as np

#: Grid growth factor when the convergence factor is still above tolerance.
GROWTH = 1.6

#: Default bound on the convergence factor (3.22).
DEFAULT_TOL = 1e-9

#: Refuse to project onto more counts than this, whatever the tolerance asks.
MAX_Y = 500_000


def promoter_stationary(Q):
    """Stationary distribution of the promoter chain on its own."""
    n = Q.shape[0]
    M = Q.T.copy()
    M[0, :] = 1.0
    rhs = np.zeros(n)
    rhs[0] = 1.0
    return np.linalg.solve(M, rhs)


def factorial_moments(Q, act, k_y, gamma, order=2):
    """Factorial moments of y, exactly and without any truncation.

    Expanding the generating function about z = 1 as sum_n c_n u^n and matching
    powers gives (n gamma I - Q^T) c_n = K c_{n-1} with c_0 the promoter's
    stationary law, so E[y(y-1)...(y-j+1)] = j! sum_i c_j[i]. Independent of the
    projection, which makes it a useful check on it.
    """
    QT = Q.T
    K = np.diag(k_y * np.asarray(act, dtype=float))
    eye = np.eye(Q.shape[0])

    c = promoter_stationary(Q)
    out, factorial = [], 1.0
    for j in range(1, order + 1):
        c = np.linalg.solve(j * gamma * eye - QT, K @ c)
        factorial *= j
        out.append(factorial * c.sum())
    return out


def moments(Q, act, k_y, gamma):
    """(mean, variance, Fano factor) of the stationary count, exactly."""
    m1, m2 = factorial_moments(Q, act, k_y, gamma, order=2)
    var = m2 + m1 - m1**2
    return m1, var, var / m1 if m1 > 0 else np.nan


def drift_constants(act, k_y, gamma):
    """Foster-Lyapunov constants for V(sigma, y) = 1 + y.

    The criterion (3.17) asks for QV(x) <= C1 - C2 V(x) with V norm-like. Here
    only transcription and degradation move y, so

        QV(sigma, y) = k_y act(sigma) - gamma y <= k_y max(act) - gamma y,

    which is C1 - C2 (1 + y) for C2 = gamma and C1 = k_y max(act) + gamma. The
    promoter is bounded so it cannot spoil norm-likeness. This is what licenses
    the error bound, and it holds for every parameter value with gamma > 0.
    """
    return float(k_y * np.max(act) + gamma), float(gamma)


def _sfsp_solve(Q, act, k_y, gamma, y_max, designated):
    """Stationary distribution of the redirected chain, and its outflow rate.

    The redirection is a rank-one update, Qbar = Q_n + c_n b_l, and it collapses
    neatly. Transposing the stationary equation,

        Q_n^T pi + b_l^T (c_n^T pi) = 0   =>   Q_n^T pi = -r_out e_l,

    so pi solves the *sub*-generator against a point source at the designated
    state. Q_n is strictly substochastic and therefore invertible, and r_out is
    only an overall scale that normalisation removes -- so one solve against
    -e_l suffices, with no eigenvector problem to solve.
    """
    n = Q.shape[0]
    QT = Q.T
    K = np.diag(k_y * np.asarray(act, dtype=float))
    eye = np.eye(n)

    # Block tridiagonal in y: sub-diagonal K, diagonal QT - K - gamma y I,
    # super-diagonal gamma (y+1) I. Sweep down for pi_{y+1} = R_y pi_y. The
    # escaping transcription at y_max stays in the diagonal -- that is exactly
    # the mass sFSP redirects, rather than discarding as a reflecting cut would.
    R = np.zeros((y_max + 1, n, n))
    for y in range(y_max - 1, -1, -1):
        B = QT - K - gamma * (y + 1) * eye
        C = gamma * (y + 2) * eye
        R[y] = -np.linalg.solve(B + C @ R[y + 1], K)

    # y = 0 takes no inflow from below, and carries the point source.
    source = np.zeros(n)
    source[designated] = -1.0
    M = (QT - K) + gamma * R[0]
    P = np.zeros((y_max + 1, n))

    # The point-source form needs Q_n to be strictly substochastic, which it is
    # only while some mass actually escapes. Project far enough above the
    # distribution and the escaping rate underflows to zero, leaving M
    # numerically singular -- the solve then fails or returns noise. In that
    # limit there is nothing to redirect and the redirected chain's law is just
    # the null vector, so fall back to it. M is only as big as the promoter, so
    # the decomposition costs nothing.
    if np.linalg.cond(M) > 1e12:
        P[0] = np.linalg.svd(M)[2][-1]
    else:
        P[0] = np.linalg.solve(M, source)

    for y in range(y_max):
        P[y + 1] = R[y] @ P[y]

    total = P.sum()
    if not np.isfinite(total) or total == 0.0:
        raise FloatingPointError(
            "sFSP solve produced no usable mass; the projection is probably "
            "far below the mean for these rates"
        )
    P = P / total
    p = np.clip(P.sum(axis=1), 0.0, None)
    p = p / p.sum()

    # c_n is supported on the top of the grid: only transcription escapes.
    outflow = float(k_y * np.dot(np.asarray(act, dtype=float),
                                 np.clip(P[y_max], 0.0, None)))
    return p, outflow


def stationary_pmf(Q, act, k_y, gamma, y_max=None, tol=DEFAULT_TOL,
                   designated=0, return_diagnostics=False):
    """P(y) for y = 0..y_max by sFSP, expanding until the error bound is met.

    This is Algorithm 1 of the paper: build the redirected generator on the
    current truncation, solve it, evaluate the convergence factor, and expand
    if it is still above ``tol``.

    ``designated`` is the promoter index of the redirection target x_l, taken at
    y = 0. Putting it at the bottom of the grid keeps V(x_l) = 1, which makes
    ``||E_n||_V`` -- and so the convergence factor -- as small as it can be.

    With ``return_diagnostics``, also returns ``(r_out, gamma_factor, y_max)``:
    the outflow rate (3.15), the convergence factor (3.22), and the projection
    actually used. The convergence factor is the quantity Theorem 3.1 brackets
    the true error with, so it is worth checking rather than assuming.
    """
    if not np.all(np.isfinite(Q)) or not np.isfinite(k_y) or k_y < 0:
        raise ValueError("non-finite or negative rates")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    if y_max is None:
        mean, var, _ = moments(Q, act, k_y, gamma)
        y_max = int(mean + 10.0 * np.sqrt(max(var, 1.0)) + 12)
    y_max = max(int(y_max), 8)

    while True:
        p, outflow = _sfsp_solve(Q, act, k_y, gamma, y_max, designated)
        # ||E_n||_V = V(x_l) + max over the boundary, with V = 1 + y. The
        # boundary is the top row of the grid and x_l sits at y = 0.
        norm_En = 1.0 + (1.0 + y_max)
        factor = outflow * norm_En
        if factor <= tol or y_max >= MAX_Y:
            break
        y_max = min(int(y_max * GROWTH) + 8, MAX_Y)

    if return_diagnostics:
        return p, {"outflow": outflow, "convergence_factor": factor,
                   "y_max": y_max}
    return p


def generator_from_jumps(n_states, jumps):
    """Dense generator from ``(from, to, rate)`` triples, diagonal filled in."""
    Q = np.zeros((n_states, n_states))
    for u, v, rate in jumps:
        Q[u, v] += rate
        Q[u, u] -= rate
    return Q
