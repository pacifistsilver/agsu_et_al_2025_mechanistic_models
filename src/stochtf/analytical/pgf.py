"""Probability generating functions for the two-site promoter models.

The stationary count y is a Cox process: given the promoter path it is Poisson
with a random rate, so G(z) = E[z^y] describes it completely. Working with G
rather than simulating gets you the whole distribution instead of a couple of
summary statistics, and it is deterministic, so there is no Monte Carlo noise to
set an ABC tolerance against.

Write G_i(z) = E[z^y 1{sigma = i}] for the four promoter compartments and the
stationary master equation turns into a linear ODE in u = z - 1,

    gamma u dG/du = Q^T G + u K G,        G(0) = pi,

with Q the promoter generator, K = diag(k_y * act) the per-state transcription
rates and pi the stationary promoter distribution. Everything here comes out of
that one equation.

There are three routes, and they agree to machine precision:

telegraph_pgf / pgf
    Closed form, ADD gate only. The sites contribute additively, so the mRNA
    pool splits into two independent telegraph-modulated birth-death processes
    and G factorises into two Kummer functions. Exact and fast.

stationary_pmf
    The full distribution. ADD inverts the closed form by FFT; OR and AND have
    no closed form, so it solves the block-tridiagonal stationary CME instead.
    Same distribution, different road.

pgf_ode
    The general PGF for any driver, by integrating the ODE above. Seconds per
    call, so it is here to check the other two, not to sit in a sampler.

Factorial moments fall out of an exact recursion on the Taylor coefficients of G
about z = 1, so mean, variance and Fano need no truncation anywhere.
"""

import numpy as np
from scipy.special import hyp1f1

# Transcription activity per compartment, ordered 00, 10, 01, 11. Same as
# gates.ACT. The paper's monomer model is the ADD gate (rate scales with how
# many sites are bound); the heterodimer model is OR (one bound site is enough).
ACT = {
    "OR": np.array([0.0, 1.0, 1.0, 1.0]),
    "AND": np.array([0.0, 0.0, 0.0, 1.0]),
    "ADD": np.array([0.0, 1.0, 1.0, 2.0]),
}

# Inference model name -> promoter logic.
MODEL_GATE = {"monomer": "ADD", "heterodimer": "OR", "dimer": "OR"}


def promoter_generator(a_s, b_s, a_n, b_n):
    """Generator of the four-state promoter, compartments ordered 00, 10, 01, 11.

    Site s flips 0->1 at a_s and 1->0 at b_s, site n likewise. The sites are
    independent, so this is just a Kronecker sum of two two-state chains.
    """
    Q = np.zeros((4, 4))
    jumps = [(0, 1, a_s), (0, 2, a_n), (1, 0, b_s), (1, 3, a_n),
             (2, 0, b_n), (2, 3, a_s), (3, 1, b_n), (3, 2, b_s)]
    for u, v, r in jumps:
        Q[u, v] += r
        Q[u, u] -= r
    return Q


def stationary_promoter(Q):
    """Stationary distribution of the promoter chain alone."""
    M = Q.T.copy()
    M[0, :] = 1.0
    rhs = np.zeros(Q.shape[0])
    rhs[0] = 1.0
    return np.linalg.solve(M, rhs)


# ----------------------------------------------------------------------
# closed-form PGF: single site, and the additive (monomer) gate
# ----------------------------------------------------------------------

def telegraph_pgf(z, alpha, beta, k_y, gamma):
    """PGF of one telegraph-modulated birth-death process.

    One site switching on at alpha and off at beta, transcribing at k_y while
    on, gives the Poisson-Beta (Peccoud-Ycart) stationary law:

        G(z) = 1F1(alpha/gamma; (alpha+beta)/gamma; (k_y/gamma)(z-1)).

    Takes complex z, which is what lets us invert by FFT.
    """
    return hyp1f1(alpha / gamma, (alpha + beta) / gamma, (k_y / gamma) * (z - 1.0))


def pgf(z, a_s, b_s, a_n, b_n, k_y, gamma, gate="ADD"):
    """Stationary PGF E[z^y].

    Only ADD has a closed form. Its rate k_y*(sigma_s + sigma_n) is additive, so
    the mRNA pool splits into two independent telegraph processes and G is their
    product.

    OR and AND couple the sites (k_y*(1 - (1-sigma_s)(1-sigma_n)) doesn't
    separate) so there's no product form. Use pgf_ode for the PGF itself, or
    stationary_pmf if you just want the distribution.
    """
    if gate != "ADD":
        raise ValueError(
            f"no closed-form PGF for the {gate!r} gate; the transcription rate "
            "does not separate across the two sites. Use pgf_ode() for the PGF "
            "or stationary_pmf() for the distribution."
        )
    return (telegraph_pgf(z, a_s, b_s, k_y, gamma)
            * telegraph_pgf(z, a_n, b_n, k_y, gamma))


# ----------------------------------------------------------------------
# factorial moments: exact, truncation-free
# ----------------------------------------------------------------------

def factorial_moments(a_s, b_s, a_n, b_n, k_y, gamma, gate="OR", order=2):
    """Factorial moments E[y(y-1)...(y-j+1)] for j = 1..order.

    Expand G about z = 1 as sum_n c_n u^n, match powers in the ODE and you get
    (n gamma I - Q^T) c_n = K c_{n-1} with c_0 = pi, so

        E[y(y-1)...(y-j+1)] = j! * sum_i c_j[i].

    No series truncation, no state-space truncation. Exact.
    """
    Q = promoter_generator(a_s, b_s, a_n, b_n)
    QT = Q.T
    K = np.diag(k_y * ACT[gate])
    eye = np.eye(4)

    c = stationary_promoter(Q)
    out = []
    factorial = 1.0
    for j in range(1, order + 1):
        c = np.linalg.solve(j * gamma * eye - QT, K @ c)
        factorial *= j
        out.append(factorial * c.sum())
    return out


def moments(a_s, b_s, a_n, b_n, k_y, gamma, gate="OR"):
    """(mean, variance, Fano factor), exactly.

    This disagrees with heterodimer.fano on the OR gate. It agrees with
    gates.analytic and with the exact FSP distribution, so heterodimer.fano is
    the odd one out. See tests/test_pgf.py.
    """
    m1, m2 = factorial_moments(a_s, b_s, a_n, b_n, k_y, gamma, gate, order=2)
    mean = m1
    var = m2 + mean - mean**2
    return mean, var, var / mean


# ----------------------------------------------------------------------
# the full distribution
# ----------------------------------------------------------------------

def pmf_from_pgf(pgf_values, n_points):
    """Invert a PGF sampled on the unit circle.

    pgf_values[j] = G(exp(2 pi i j / n_points)). Since
    G(z_j) = sum_y P(y) exp(2 pi i j y / n_points), a forward FFT hands the
    coefficients straight back.
    """
    return np.real(np.fft.fft(pgf_values) / n_points)


def _pmf_block_tridiagonal(a_s, b_s, a_n, b_n, k_y, gamma, gate, y_max):
    """Stationary P(y) by backward block-tridiagonal elimination.

    The stationary CME only couples neighbouring counts,

        K P_{y-1} + (Q^T - K - gamma y I) P_y + gamma (y+1) P_{y+1} = 0,

    so it's block-tridiagonal with 4x4 blocks. Write P_{y+1} = R_y P_y, sweep
    down from R_{y_max} = 0, and you pay O(y_max) tiny solves instead of one big
    sparse one. That's about 30x faster than assembling the full matrix, which
    matters here because this runs in the sampler's inner loop.
    """
    Q = promoter_generator(a_s, b_s, a_n, b_n)
    QT = Q.T
    K = np.diag(k_y * ACT[gate])
    eye = np.eye(4)

    R = np.zeros((y_max + 1, 4, 4))
    for y in range(y_max - 1, -1, -1):
        B = QT - K - gamma * (y + 1) * eye
        C = gamma * (y + 2) * eye
        R[y] = -np.linalg.solve(B + C @ R[y + 1], K)

    # y = 0 has no inflow from below: (Q^T - K + gamma R_0) P_0 = 0.
    M = (QT - K) + gamma * R[0]
    _, _, vt = np.linalg.svd(M)
    P = np.zeros((y_max + 1, 4))
    P[0] = vt[-1]
    for y in range(y_max):
        P[y + 1] = R[y] @ P[y]

    p = P.sum(axis=1)
    # Sign of the null vector is arbitrary, so flip it to carry positive mass.
    if p.sum() < 0:
        p = -p
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError(
            "stationary solve produced non-positive mass; y_max is probably "
            "far below the mean for these rates"
        )
    return np.clip(p, 0.0, None) / total


# Past roughly this k_y/gamma, FFT inversion of the closed form stops being
# trustworthy. hyp1f1 gets called at (k_y/gamma)(z-1), whose modulus hits
# 2 k_y/gamma on the unit circle, and it bleeds precision as that grows
# (~1e-7 at 30, ~2e-4 at 250) before returning NaN somewhere around 500. The
# CME route doesn't have this problem.
PGF_FFT_MAX_K_OVER_GAMMA = 20.0


def stationary_pmf(a_s, b_s, a_n, b_n, k_y, gamma, gate="OR", y_max=None,
                   method="cme", n_fft=None):
    """Full stationary distribution P(y) for y = 0..y_max.

    method="cme" (the default) solves the block-tridiagonal stationary CME.
    Exact for every gate and stable at any k_y/gamma, which is what the real
    counts need: they run to ~1000 molecules.

    method="pgf" inverts the closed form by FFT. ADD gate only, agrees with the
    CME route to machine precision at small k_y/gamma, and is the independent
    cross-check in tests/test_pgf.py. Past PGF_FFT_MAX_K_OVER_GAMMA it degrades,
    so it raises instead of quietly handing back wrong numbers.

    y_max defaults to twelve standard deviations above the mean, and
    factorial_moments gives both of those exactly without solving anything.
    """
    if y_max is None:
        mean, second = factorial_moments(a_s, b_s, a_n, b_n, k_y, gamma, gate,
                                         order=2)
        var = second + mean - mean**2
        y_max = max(32, int(mean + 12.0 * np.sqrt(max(var, 1.0)) + 12))

    if method == "cme":
        return _pmf_block_tridiagonal(a_s, b_s, a_n, b_n, k_y, gamma, gate, y_max)

    if method != "pgf":
        raise ValueError(f"method must be 'cme' or 'pgf', got {method!r}")

    if gate != "ADD":
        raise ValueError(
            f"method='pgf' needs a closed-form PGF, which only the ADD gate "
            f"has; got gate={gate!r}. Use method='cme'."
        )
    ratio = k_y / gamma
    if ratio > PGF_FFT_MAX_K_OVER_GAMMA:
        raise ValueError(
            f"k_y/gamma = {ratio:.3g} exceeds {PGF_FFT_MAX_K_OVER_GAMMA:g}, "
            "beyond which hyp1f1 on the unit circle loses precision and "
            "eventually returns NaN. Use method='cme'."
        )

    # FFT length must exceed y_max, otherwise the tail wraps onto the head.
    if n_fft is None:
        n_fft = 1 << int(np.ceil(np.log2(max(2 * (y_max + 1), 64))))
    z = np.exp(2j * np.pi * np.arange(n_fft) / n_fft)
    p = pmf_from_pgf(pgf(z, a_s, b_s, a_n, b_n, k_y, gamma, "ADD"), n_fft)
    p = np.clip(p[:y_max + 1], 0.0, None)
    return p / p.sum()


def model_pmf(model, a_s, b_s, a_n, b_n, k_y, gamma, y_max=None):
    """Stationary distribution for a named inference model ('monomer'/'dimer')."""
    try:
        gate = MODEL_GATE[model]
    except KeyError:
        raise ValueError(
            f"unknown model {model!r}; expected one of {sorted(MODEL_GATE)}"
        ) from None
    return stationary_pmf(a_s, b_s, a_n, b_n, k_y, gamma, gate, y_max=y_max)


# ----------------------------------------------------------------------
# general PGF by ODE: the reference implementation
# ----------------------------------------------------------------------

def pgf_ode(u, a_s, b_s, a_n, b_n, k_y, gamma, gate="OR",
            n_steps=4000, s_min=-14.0):
    """G(1+u) for arbitrary u, by integrating the defining ODE.

    Substituting u = U e^s kills the regular singular point at u = 0 and leaves
    dG/ds = [Q^T G + e^s U K G]/gamma, which we integrate from s = s_min (where
    G is still pi to rounding) up to s = 0.

    Works for any gate but costs seconds a call. It's here to check the closed
    form and the CME solve, not to be used in a sampler.
    """
    U = np.atleast_1d(u).astype(complex)
    Q = promoter_generator(a_s, b_s, a_n, b_n)
    QT = Q.T.astype(complex)
    K = np.diag(k_y * ACT[gate]).astype(complex)

    G = np.tile(stationary_promoter(Q).astype(complex), (U.size, 1))
    s = np.linspace(s_min, 0.0, n_steps + 1)
    h = s[1] - s[0]

    def deriv(sv, Gv):
        return (Gv @ QT.T + (np.exp(sv) * U)[:, None] * (Gv @ K.T)) / gamma

    for i in range(n_steps):
        sv = s[i]
        k1 = deriv(sv, G)
        k2 = deriv(sv + h / 2, G + h / 2 * k1)
        k3 = deriv(sv + h / 2, G + h / 2 * k2)
        k4 = deriv(sv + h, G + h * k3)
        G = G + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return G.sum(axis=1)
