import numpy as np


def t_on_correct(a_s, b_s, a_n, b_n):
    """Mean ON (burst) duration: entry-weighted MFPT from {10,01} to 00."""
    p_00 = (b_s / (a_s + b_s)) * (b_n / (a_n + b_n))
    return (1.0 - p_00) / (p_00 * (a_s + a_n))


def t_on_linalg(a_s, b_s, a_n, b_n):
    """Same thing by direct linear solve, for cross-checking."""
    R = np.array([b_s + a_n, b_n + a_s, b_s + b_n])   # states 10, 01, 11
    P = np.zeros((3, 3))
    P[0, 2] = a_n / R[0]
    P[1, 2] = a_s / R[1]
    P[2, 0] = b_n / R[2]
    P[2, 1] = b_s / R[2]
    T = np.linalg.solve(np.eye(3) - P, 1.0 / R)
    ent = np.array([a_s, a_n, 0.0])
    return (ent @ T) / ent.sum()


def burst_size_correct(a_s, b_s, a_n, b_n, k_y):
    return k_y * t_on_correct(a_s, b_s, a_n, b_n)


def burst_frequency_correct(a_s, b_s, a_n, b_n):
    return 1.0 / (t_on_correct(a_s, b_s, a_n, b_n) + 1.0 / (a_s + a_n))