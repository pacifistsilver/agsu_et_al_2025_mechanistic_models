"""Monomer model
"""
import numpy as np

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


def occ(a, b):
    """Returns the bound fraction p and the free fraction q."""
    return a/(a+b), b/(a+b)

def t_off(a_s, b_s, a_n, b_n):
    """Returns the mean OFF duration, exactly 1/(alpha_s + alpha_n)."""
    return (1 / (a_s + a_n))

def t_on(a_s, b_s, a_n, b_n):
    return ((a_n * b_s**2) + (a_s * b_n**2)) / ((b_s * b_n) * ((a_n * b_s) + (a_s * b_n)))

def burst_frequency(a_s, b_s, a_n, b_n):
    """Returns the burst frequency, f = 1 / (<T_on> + <T_off>)."""
    tau_on = t_on(a_s, b_s, a_n, b_n)
    tau_off = t_off(a_s, b_s, a_n, b_n)
    return 1 / (tau_on + tau_off)


def burst_size(a_s, b_s, a_n, b_n, k_y):
    """Returns the mean burst size, b = k_y <T_on>."""
    return k_y * t_on(a_s, b_s, a_n, b_n)


print(t_on(10, 1, 0.1, 0.1))