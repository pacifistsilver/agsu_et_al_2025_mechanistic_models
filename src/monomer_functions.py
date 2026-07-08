import numpy as np
import scipy
import scipy.integrate
from numpy import matlib


def calculate_target_fano(
    alpha_n=0.3, alpha_s=0.5, beta_n=0.2, beta_s=0.06, N_N=10, N_S=2, N_tot=9, S_tot=1
):
    b_y = np.random.geometric(p=1 / 20)
    p_s = (alpha_s * N_S) / (alpha_s * N_S + beta_s)
    var_s = S_tot * p_s * (1.0 - p_s)

    p_n = (alpha_n * N_N) / (alpha_n * N_N + beta_n)
    var_n = N_tot * p_n * (1.0 - p_n)

    mean_fy = k_y0 + (k_ys * S_tot * p_s) + (k_yn * N_tot * p_n)

    intrinsic = 1.0 + b_y

    ext_s = (b_y / mean_fy) * ((k_ys**2 * var_s) / (gamma_y + alpha_s * N_S + beta_s))
    ext_n = (b_y / mean_fy) * ((k_yn**2 * var_n) / (gamma_y + alpha_n * N_N + beta_n))

    return intrinsic + ext_s + ext_n
