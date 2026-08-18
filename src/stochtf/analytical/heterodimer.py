"""
Heterodimer model with mRNA production/degradation.

Promoter states (sigma_s, sigma_n) in {0,1}^2, mapping to the master-equation
compartments n00, n10, n01, n11.  Site s flips 0->1 at alpha_s, 1->0 at beta_s;
site n likewise with (alpha_n, beta_n).  The two sites are independent.
mRNA y is produced at rate k_y whenever *at least one* site is bound
(states 10, 01, 11) and degrades at rate gamma per molecule.

Because the silent set is the single state 00, visits to it are regeneration
points: ON/OFF durations are i.i.d. and the burst decomposition is EXACT.

Everything below is for one gene copy; for N copies multiply the burst
frequency by N (burst size and Fano factor are N-independent).
"""

import numpy as np

# alpha = k [sox2:nanog]
# p00 -> p10, p00 -> p01

def derived(a_s: np.float64, b_s: np.float64, a_n: np.float64, b_n: np.float64) -> tuple:
    """Returns individual probabilities of bound and unbound SOX2/NANOG states.
    Args:
        a_s: concentration dependent rate of [SOX2]. np.float64 type.
        b_s: affinity of SOX2 for chromatin. np.float64 type.
        a_n: concentration dependent rate of [NANOG]. np.float64 type.
        b_n: affinity of NANOG for chromatin. np.float64 type
    Return:
        Tuple composed of six items consisting of.
        lambda_s, lambda_n, PS=1, PN=1, PS=0, PN=0
        where Pi refers to the individual probabilities of finding either SOX2 or NANOG
        bind on their own. 
        """
    lam_s, lam_n = a_s + b_s, a_n + b_n
    p_s1, p_n1 = a_s / lam_s, a_n / lam_n
    p_s0 = b_s / lam_s
    p_n0 = b_n / lam_n
    return lam_s, lam_n, p_s1, p_n1, p_s0, p_n0


def p00(a_s_hat, b_s, a_n_hat, b_n):
    """Calculates probability of the P00 state which is the product of PS=0 PN=0. i.e. P_00 = PS=0 * PN=0.
    Under the assumption of independent binding. 

    Args:
        a_s: concentration dependent rate of [SOX2]. np.float64 type.
        b_s: affinity of SOX2 for chromatin. np.float64 type.
        a_n: concentration dependent rate of [NANOG]. np.float64 type.
        b_n: affinity of NANOG for chromatin. np.float64 type
    Return:
        Probability of finding both PN=0 and PS=0. i.e. the product of PS=0 * PN=0
    """
    return (b_s / (a_s_hat + b_s)) * (b_n / (a_n_hat + b_n))


def p01(a_s, b_s, a_n, b_n):
    """Calculates probability of the P01 state which is the product of PS=0 PN=1. i.e. P_01 = PS=0 * PN=1.
    Under the assumption of independent binding. 

    Args:
        a_s: concentration dependent rate of [SOX2]. np.float64 type.
        b_s: affinity of SOX2 for chromatin. np.float64 type.
        a_n: concentration dependent rate of [NANOG]. np.float64 type.
        b_n: affinity of NANOG for chromatin. np.float64 type
    Return:
        Probability of finding both PN=1 and PS=0. i.e. the product of PS=0 * PN=1
    """
    return (b_s / (a_s + b_s)) * (a_n / (a_n + b_n))


def p10(a_s, b_s, a_n, b_n):
    """Calculates probability of the P10 state which is the product of PS=1 PN=0. i.e. P_10 = PS=1 * PN=0.
    Under the assumption of independent binding. 

    Args:
        a_s: concentration dependent rate of [SOX2]. np.float64 type.
        b_s: affinity of SOX2 for chromatin. np.float64 type.
        a_n: concentration dependent rate of [NANOG]. np.float64 type.
        b_n: affinity of NANOG for chromatin. np.float64 type
    Return:
        Probability of finding both PN=0 and PS=1. i.e. the product of PS=1 * PN=0
    """
    return (a_s / (a_s + b_s)) * (b_n / (a_n + b_n))


def p11(a_s, b_s, a_n, b_n):
    """Calculates probability of the P11 state which is the product of PS=1 PN=1. i.e. P_11 = PS=1 * PN=1.
    Under the assumption of independent binding. 

    Args:
        a_s: concentration dependent rate of [SOX2]. np.float64 type.
        b_s: affinity of SOX2 for chromatin. np.float64 type.
        a_n: concentration dependent rate of [NANOG]. np.float64 type.
        b_n: affinity of NANOG for chromatin. np.float64 type
    Return:
        Probability of finding both PN=1 and PS=1. i.e. the product of PS=1 * PN=1
    """
    return (a_s / (a_s + b_s)) * ((a_n / (a_n + b_n)))


def pbound(a_s, b_s, a_n, b_n):
    """Calculates the probability over all bound states. i.e. P10, P01, P11. 
    
    """
    return 1 - ((b_s * b_n) / ((a_s + b_s) * (a_n + b_n)))


def tbound(a_s, b_s, a_n, b_n):
    """Returns the MFPT (Mean First Passage Time) of the heterodimer in a state averaged over all bound states. i.e. P10, P10, P11.
    
    Args:
        a_s: concentration dependent rate of [SOX2]. np.float64 type.
        b_s: affinity of SOX2 for chromatin. np.float64 type.
        a_n: concentration dependent rate of [NANOG]. np.float64 type.
        b_n: affinity of NANOG for chromatin. np.float64 type
    Returns:
        The MFPT averaged over all bound states. 
    """
    _, _, p_s1, p_n1, p_s0, p_n0 = derived(a_s, b_s, a_n, b_n)
    p_bound = pbound(a_s, b_s, a_n, b_n)
    num = ((b_n**2) * (p_s1 * p_s0)) + ((b_s**2) * (p_n1 * p_n0)) + (b_s * b_n * p_bound)
    bot = (b_s * b_n * p_bound) * ((b_n * p_s0) + (b_s * p_n0))
    return num / bot



def t_off(a_s, b_s, a_n, b_n):
    """Mean OFF (silent) duration.  Exactly exponential: 1/(alpha_s+alpha_n)."""
    return 1.0 / (a_s + a_n)


def t_on(a_s, b_s, a_n, b_n):
    """Mean ON duration = mean excursion time from 00 back to 00, minus the OFF part"""
    MFPT = tbound(a_s, b_s, a_n, b_n)
    return MFPT


def burst_frequency(a_s, b_s, a_n, b_n):
    """Return burst frequency. f = 1/(<T_on>+<T_off>)"""
    tau_on = t_on(a_s, b_s, a_n, b_n)
    tau_off = t_off(a_s, b_s, a_n, b_n)
    return 1 / (tau_on + tau_off)


def burst_size(a_s, b_s, a_n, b_n, k_y):
    """Return burst size. b = k_y * <T_on>."""
    return k_y * t_on(a_s, b_s, a_n, b_n)


def mean_y(a_s, b_s, a_n, b_n, k_y, gamma, N=1):
    """<y> = N k_y (1 - q_s q_n)/gamma  = N b f / gamma."""
    _, _, _, _, p_s0, p_n0 = derived(a_s, b_s, a_n, b_n)
    return N * k_y * (1.0 - p_s0 * p_n0) / gamma


def fano(a_s, b_s, a_n, b_n, k_y, gamma):
    """Exact Fano factor (N-independent)."""
    lam_s, lam_n, p_s, p_n, q_s, q_n = derived(a_s, b_s, a_n, b_n)
    bracket = (
        q_n**2 * p_s * q_s / (gamma + lam_s)
        + q_s**2 * p_n * q_n / (gamma + lam_n)
        + p_s * q_s * p_n * q_n / (gamma + lam_s + lam_n)
    )
    return 1.0 + k_y * bracket / mean_y(a_s, b_s, a_n, b_n, k_y, gamma, N=1)


def _phase_type(a_s, b_s, a_n, b_n, k_y):
    """Build the embedded emission chain on transient states [11, 10, 01].

    Each step is either a production event (emit 1, stay) or a promoter jump.
        A[i,i] = k_y / (k_y + R_i)                 emit-and-stay
        B[i,j] = rate(i->j) / (k_y + R_i)          jump, transient -> transient
        t[i]   = rate(i->00) / (k_y + R_i)         absorption
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
    """Exact P(burst size = m) for m = 0..mmax.
    P(N=m) = ent . [(I-B)^-1 A]^m . (I-B)^-1 t"""
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
    """Eigenvalues of U = decay ratios of the three geometric components."""
    A, B, t, ent = _phase_type(a_s, b_s, a_n, b_n, k_y)
    U = np.linalg.inv(np.eye(3) - B) @ A
    return np.sort(np.real(np.linalg.eigvals(U)))[::-1]


def burst_size_pgf(w, a_s, b_s, a_n, b_n, k_y):
    """Closed-form PGF B(w) of the burst size (for cross-checking the PMF)."""
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
    """Simulate the promoter, recording ON durations and molecules made per burst."""
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
    """Stationary distribution over (promoter, y) for a single copy."""
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
