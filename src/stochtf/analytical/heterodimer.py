"""Heterodimer promoter with mRNA production and degradation.
"""

import numpy as np

# alpha = k [sox2:nanog]
# p00 -> p10, p00 -> p01

def derived(a_s: np.float64, b_s: np.float64, a_n: np.float64, b_n: np.float64) -> tuple:
    """Returns the per-site switching rates and occupancies.

    Args:
        a_s: SOX2 binding rate, concentration dependent.
        b_s: SOX2 unbinding rate.
        a_n: NANOG binding rate, concentration dependent.
        b_n: NANOG unbinding rate.

    Returns:
        A tuple (lambda_s, lambda_n, p_s, p_n, q_s, q_n): the total
        switching rate of each site, then the bound and free probabilities
        of each, treating the sites on their own.
    """
    lam_s, lam_n = a_s + b_s, a_n + b_n
    p_s1, p_n1 = a_s / lam_s, a_n / lam_n
    p_s0 = b_s / lam_s
    p_n0 = b_n / lam_n
    return lam_s, lam_n, p_s1, p_n1, p_s0, p_n0


def p00(a_s_hat, b_s, a_n_hat, b_n):
    """Returns P(00): SOX2 free, NANOG free.

    The sites are independent, so this is the product of the two
    single-site probabilities.

    Args:
        a_s: SOX2 binding rate, concentration dependent.
        b_s: SOX2 unbinding rate.
        a_n: NANOG binding rate, concentration dependent.
        b_n: NANOG unbinding rate.

    Returns:
        The stationary probability of state 00.
    """
    return (b_s / (a_s_hat + b_s)) * (b_n / (a_n_hat + b_n))


def p01(a_s, b_s, a_n, b_n):
    """Returns P(01): SOX2 free, NANOG bound.

    The sites are independent, so this is the product of the two
    single-site probabilities.

    Args:
        a_s: SOX2 binding rate, concentration dependent.
        b_s: SOX2 unbinding rate.
        a_n: NANOG binding rate, concentration dependent.
        b_n: NANOG unbinding rate.

    Returns:
        The stationary probability of state 01.
    """
    return (b_s / (a_s + b_s)) * (a_n / (a_n + b_n))


def p10(a_s, b_s, a_n, b_n):
    """Returns P(10): SOX2 bound, NANOG free.

    The sites are independent, so this is the product of the two
    single-site probabilities.

    Args:
        a_s: SOX2 binding rate, concentration dependent.
        b_s: SOX2 unbinding rate.
        a_n: NANOG binding rate, concentration dependent.
        b_n: NANOG unbinding rate.

    Returns:
        The stationary probability of state 10.
    """
    return (a_s / (a_s + b_s)) * (b_n / (a_n + b_n))


def p11(a_s, b_s, a_n, b_n):
    """Returns P(11): SOX2 bound, NANOG bound.

    The sites are independent, so this is the product of the two
    single-site probabilities.

    Args:
        a_s: SOX2 binding rate, concentration dependent.
        b_s: SOX2 unbinding rate.
        a_n: NANOG binding rate, concentration dependent.
        b_n: NANOG unbinding rate.

    Returns:
        The stationary probability of state 11.
    """
    return (a_s / (a_s + b_s)) * ((a_n / (a_n + b_n)))


def pbound(a_s, b_s, a_n, b_n):
    """Returns P(at least one site bound), i.e. P10 + P01 + P11.

    Args:
        a_s: SOX2 binding rate, concentration dependent.
        b_s: SOX2 unbinding rate.
        a_n: NANOG binding rate, concentration dependent.
        b_n: NANOG unbinding rate.

    Returns:
        The stationary probability that the promoter is active.
    """
    return 1 - ((b_s * b_n) / ((a_s + b_s) * (a_n + b_n)))


def tbound(a_s, b_s, a_n, b_n):
    """Returns the mean first-passage time out of the bound states.

    Averaged over the bound states 10, 01 and 11.

    Args:
        a_s: SOX2 binding rate, concentration dependent.
        b_s: SOX2 unbinding rate.
        a_n: NANOG binding rate, concentration dependent.
        b_n: NANOG unbinding rate.

    Returns:
        The mean first-passage time back to the silent state 00.
    """
    _, _, p_s1, p_n1, p_s0, p_n0 = derived(a_s, b_s, a_n, b_n)
    p_bound = pbound(a_s, b_s, a_n, b_n)
    num = ((b_n**2) * (p_s1 * p_s0)) + ((b_s**2) * (p_n1 * p_n0)) + (b_s * b_n * p_bound)
    bot = (b_s * b_n * p_bound) * ((b_n * p_s0) + (b_s * p_n0))
    return num / bot



def t_off(a_s, b_s, a_n, b_n):
    """Returns the mean OFF duration, exactly 1/(alpha_s + alpha_n)."""
    return 1.0 / (a_s + a_n)


def t_on(a_s, b_s, a_n, b_n):
    """Returns the mean ON duration.

    The mean excursion time from state 00 back to 00, less the OFF part.

    Args:
        a_s: SOX2 binding rate.
        b_s: SOX2 unbinding rate.
        a_n: NANOG binding rate.
        b_n: NANOG unbinding rate.

    Returns:
        The mean time the promoter spends active per burst.
    """
    MFPT = tbound(a_s, b_s, a_n, b_n)
    return MFPT


def burst_frequency(a_s, b_s, a_n, b_n):
    """Returns the burst frequency, f = 1 / (<T_on> + <T_off>)."""
    tau_on = t_on(a_s, b_s, a_n, b_n)
    tau_off = t_off(a_s, b_s, a_n, b_n)
    return 1 / (tau_on + tau_off)


def burst_size(a_s, b_s, a_n, b_n, k_y):
    """Returns the mean burst size, b = k_y <T_on>."""
    return k_y * t_on(a_s, b_s, a_n, b_n)


def mean_y(a_s, b_s, a_n, b_n, k_y, gamma, N=1):
    """Returns the mean count, <y> = N k_y (1 - q_s q_n) / gamma."""
    _, _, _, _, p_s0, p_n0 = derived(a_s, b_s, a_n, b_n)
    return N * k_y * (1.0 - p_s0 * p_n0) / gamma


def fano(a_s, b_s, a_n, b_n, k_y, gamma):
    """Returns the exact Fano factor, which is independent of N."""
    lam_s, lam_n, p_s, p_n, q_s, q_n = derived(a_s, b_s, a_n, b_n)
    bracket = (
        q_n**2 * p_s * q_s / (gamma + lam_s)
        + q_s**2 * p_n * q_n / (gamma + lam_n)
        + p_s * q_s * p_n * q_n / (gamma + lam_s + lam_n)
    )
    return 1.0 + k_y * bracket / mean_y(a_s, b_s, a_n, b_n, k_y, gamma, N=1)


def _phase_type(a_s, b_s, a_n, b_n, k_y):
    """Builds the embedded emission chain on transient states 11, 10, 01.

    Each step is either a production event (emit one, stay) or a promoter
    jump::

        A[i,i] = k_y / (k_y + R_i)            emit and stay
        B[i,j] = rate(i->j) / (k_y + R_i)     jump, transient -> transient
        t[i]   = rate(i->00) / (k_y + R_i)    absorption

    Args:
        a_s: SOX2 binding rate.
        b_s: SOX2 unbinding rate.
        a_n: NANOG binding rate.
        b_n: NANOG unbinding rate.
        k_y: Transcription rate in the active states.

    Returns:
        A tuple (A, B, t, entry) defining the emission chain.
    """
    R = np.array([b_s + b_n, b_s + a_n, a_s + b_n])  # promoter exit rates
    tot = k_y + R

    B = np.zeros((3, 3))
    B[0, 2] = b_s / tot[0]  # 11 -> 01   (s unbinds)
    B[0, 1] = b_n / tot[0]  # 11 -> 10   (n unbinds)
    B[1, 0] = a_n / tot[1]  # 10 -> 11   (n rebinds)
    B[2, 0] = a_s / tot[2]  # 01 -> 11   (s rebinds)

    t = np.array([0.0, b_s / tot[1], b_n / tot[2]])  # -> 00, burst ends
    A = np.diag(k_y / tot)

    # entry distribution: from 00, bind s (rate a_s) -> 10 ; bind n -> 01
    ent = np.array([0.0, a_s / (a_s + a_n), a_n / (a_s + a_n)])
    return A, B, t, ent


def burst_size_pmf(a_s, b_s, a_n, b_n, k_y, mmax=200):
    """Returns the exact burst-size distribution for m = 0..mmax.

    P(N=m) = ent . [(I-B)^-1 A]^m . (I-B)^-1 t

    Args:
        a_s: SOX2 binding rate.
        b_s: SOX2 unbinding rate.
        a_n: NANOG binding rate.
        b_n: NANOG unbinding rate.
        k_y: Transcription rate in the active states.
        mmax: Largest burst size to evaluate.

    Returns:
        P(burst size = m) for m = 0..mmax.
    """
    A, B, t, ent = _phase_type(a_s, b_s, a_n, b_n, k_y)
    S = np.linalg.inv(np.eye(3) - B)
    U = S @ A  # kernel: state at the next emission
    v = S @ t  # absorb before any further emission

    pmf = np.empty(mmax + 1)
    row = ent.copy()
    for m in range(mmax + 1):
        pmf[m] = row @ v
        row = row @ U
    return pmf


def burst_size_geometric_rates(a_s, b_s, a_n, b_n, k_y):
    """Returns the decay ratios of the three geometric components.

    These are the eigenvalues of U.

    Args:
        a_s: SOX2 binding rate.
        b_s: SOX2 unbinding rate.
        a_n: NANOG binding rate.
        b_n: NANOG unbinding rate.
        k_y: Transcription rate in the active states.

    Returns:
        The three decay ratios, largest first.
    """
    A, B, t, ent = _phase_type(a_s, b_s, a_n, b_n, k_y)
    U = np.linalg.inv(np.eye(3) - B) @ A
    return np.sort(np.real(np.linalg.eigvals(U)))[::-1]


def burst_size_pgf(w, a_s, b_s, a_n, b_n, k_y):
    """Returns the closed-form burst-size PGF B(w), for checking the PMF."""
    th = k_y * (1.0 - w)
    a = th + b_s + a_n
    c = th + a_s + b_n
    d = th + b_s + b_n
    phi11 = b_s * b_n * (a + c) / (d * a * c - a_s * b_s * a - a_n * b_n * c)
    phi10 = (b_s + a_n * phi11) / a
    phi01 = (b_n + a_s * phi11) / c
    return (a_s * phi10 + a_n * phi01) / (a_s + a_n)


# ----------------------------------------------------------------------
# 4. Gillespie validation of the burst statistics
# ----------------------------------------------------------------------


def gillespie_bursts(a_s, b_s, a_n, b_n, k_y, n_bursts=20000, seed=0):
    """Simulates the promoter, recording ON durations and burst sizes."""
    rng = np.random.default_rng(seed)
    ss = sn = 0  # start silent
    durations = np.empty(n_bursts)
    sizes = np.empty(n_bursts, dtype=np.int64)
    off_times = np.empty(n_bursts)

    nb = 0
    t_start = 0.0
    off_start = 0.0
    count = 0
    t = 0.0
    while nb < n_bursts:
        on = ss or sn
        rates = np.array(
            [a_s if ss == 0 else b_s, a_n if sn == 0 else b_n, k_y if on else 0.0]
        )
        tot = rates.sum()
        t += rng.exponential(1.0 / tot)
        j = rng.choice(3, p=rates / tot)
        if j == 2:
            count += 1
            continue
        was_on = on
        if j == 0:
            ss ^= 1
        else:
            sn ^= 1
        now_on = ss or sn
        if not was_on and now_on:  # 00 -> ON : burst starts
            t_start = t
            off_times[nb] = t - off_start
            count = 0
        elif was_on and not now_on:  # ON -> 00 : burst ends
            durations[nb] = t - t_start
            sizes[nb] = count
            off_start = t
            nb += 1
    return durations, sizes, off_times


# ----------------------------------------------------------------------
# 5. exact stationary P(y) by finite state projection (validates the Fano)
# ----------------------------------------------------------------------


def fsp_stationary(a_s, b_s, a_n, b_n, k_y, gamma, ymax=400):
    """Returns the stationary distribution over (promoter, y) for one copy."""
    from scipy.sparse import lil_matrix, csc_matrix
    from scipy.sparse.linalg import spsolve

    ns, ny = 4, ymax + 1  # promoter order: 00,10,01,11
    idx = lambda s, y: s * ny + y
    act = [0, 1, 1, 1]  # sigma = OR
    jumps = [
        (0, 1, a_s),
        (0, 2, a_n),
        (1, 0, b_s),
        (1, 3, a_n),
        (2, 0, b_n),
        (2, 3, a_s),
        (3, 1, b_n),
        (3, 2, b_s),
    ]

    Q = lil_matrix((ns * ny, ns * ny))
    for s in range(ns):
        for y in range(ny):
            i = idx(s, y)
            for u, v, r in jumps:
                if u == s:
                    Q[i, idx(v, y)] += r
                    Q[i, i] -= r
            if act[s] and y < ny - 1:
                Q[i, idx(s, y + 1)] += k_y
                Q[i, i] -= k_y
            if y > 0:
                Q[i, idx(s, y - 1)] += gamma * y
                Q[i, i] -= gamma * y

    M = csc_matrix(Q).T.tolil()
    M[0, :] = 1.0  # replace one row by normalisation
    rhs = np.zeros(ns * ny)
    rhs[0] = 1.0
    pi = spsolve(csc_matrix(M), rhs)
    return pi.reshape(ns, ny).sum(axis=0)
