
"""
Burst frequency and burst size for the monomer (M) model.
 
Two emission readings are compared:
  OR  : propensity k_y (n_b + s_b)   -- as written in the CME, Eq. (20)
  AND : propensity k_y  n_b s_b      -- as written in the schematic, Eq. (3)
 
Rates from Table 3 (SMT-derived, corrected):
  beta_s = 1/15 s^-1,  beta_n = 1/4.2 s^-1,  gamma = 3e-5 s^-1,  k_y = 0.01 s^-1
"""
import numpy as np

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


def occ(a, b):
    """p = bound fraction, q = free fraction."""
    return a/(a+b), b/(a+b)

def t_off(a_s, b_s, a_n, b_n):
    """Mean OFF (silent) duration.  Exactly exponential: 1/(alpha_s+alpha_n)."""
    return (1 / (a_s + a_n))

def t_on(a_s, b_s, a_n, b_n):
    return ((a_n * b_s**2) + (a_s * b_n**2)) / ((b_s * b_n) * ((a_n * b_s) + (a_s * b_n)))

def burst_frequency(a_s, b_s, a_n, b_n):
    """Return burst frequency. f = 1/(<T_on>+<T_off>)"""
    tau_on = t_on(a_s, b_s, a_n, b_n)
    tau_off = t_off(a_s, b_s, a_n, b_n)
    return 1 / (tau_on + tau_off)


def burst_size(a_s, b_s, a_n, b_n, k_y):
    """Return burst size. b = k_y * <T_on>."""
    return k_y * t_on(a_s, b_s, a_n, b_n)


print(t_on(10, 1, 0.1, 0.1))