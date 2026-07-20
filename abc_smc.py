import os
import sys
import numpy as np
import pymc as pm
import polars as pl
import arviz as az
from numba import njit

sys.path.append(os.path.abspath('./src'))
from ssa_params import monomer_params
params, initial_state, stoichiometry = monomer_params


@njit
def fast_ssa_monomer(alpha_s, beta_s, alpha_n, beta_n, k_y, gamma_y, t_max):
    state = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    t = 0.0
    
    while t < t_max:
        nf, sf, nb, sb, y = state
        
        p0 = alpha_s * sf
        p1 = beta_s * sb
        p2 = alpha_n * nf
        p3 = beta_n * nb
        p4 = k_y * (nb + sb)
        p5 = gamma_y * y
        
        a0 = p0 + p1 + p2 + p3 + p4 + p5
        
        if a0 <= 0:
            break
            
        r1 = np.random.rand()
        r2 = np.random.rand()
        
        tau = -np.log(r1) / a0
        t += tau
        val = r2 * a0
        if val < p0:
            state[1] -= 1.0  # sf
            state[3] += 1.0   # sb
        elif val < p0 + p1:
            state[1] += 1.0   # sf
            state[3] -= 1.0   # sb
        elif val < p0 + p1 + p2:
            state[0] -= 1.0   # nf
            state[2] += 1.0   # nb
        elif val < p0 + p1 + p2 + p3:
            state[0] += 1.0   # nf
            state[2] -= 1.0   # nb
        elif val < p0 + p1 + p2 + p3 + p4:
            state[4] += 1.0   # y
        else:
            state[4] -= 1.0   # y
            
    return state[4]

@njit
def fast_ssa_dimer(alpha_s, beta_s, alpha_n, beta_n, k_y, gamma_y, t_max):
    # state = [n00, n10, n01, n11, y]
    # Initially 1 free promoter (n00), all others 0.
    state = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    t = 0.0
    
    while t < t_max:
        n00, n10, n01, n11, y = state
        
        # Calculate the 10 propensities for the heterodimer
        p0 = alpha_s * n00    # bind S to empty
        p1 = alpha_n * n00    # bind N to empty
        p2 = beta_s * n10     # unbind S from n10
        p3 = beta_n * n01     # unbind N from n01
        p4 = alpha_n * n10    # bind N to n10 -> n11
        p5 = alpha_s * n01    # bind S to n01 -> n11
        p6 = beta_n * n11     # unbind N from n11 -> n10
        p7 = beta_s * n11     # unbind S from n11 -> n01
        p8 = k_y * n11        # transcription (only from n11!)
        p9 = gamma_y * y      # degradation
        
        a0 = p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        
        if a0 <= 0:
            break
            
        r1 = np.random.rand()
        r2 = np.random.rand()
        
        tau = -np.log(r1) / a0
        t += tau
        val = r2 * a0
        
        # Determine which reaction fired
        if val < p0:
            state[0] -= 1.0  # n00
            state[1] += 1.0  # n10
        elif val < p0 + p1:
            state[0] -= 1.0  # n00
            state[2] += 1.0  # n01
        elif val < p0 + p1 + p2:
            state[1] -= 1.0  # n10
            state[0] += 1.0  # n00
        elif val < p0 + p1 + p2 + p3:
            state[2] -= 1.0  # n01
            state[0] += 1.0  # n00
        elif val < p0 + p1 + p2 + p3 + p4:
            state[1] -= 1.0  # n10
            state[3] += 1.0  # n11
        elif val < p0 + p1 + p2 + p3 + p4 + p5:
            state[2] -= 1.0  # n01
            state[3] += 1.0  # n11
        elif val < p0 + p1 + p2 + p3 + p4 + p5 + p6:
            state[3] -= 1.0  # n11
            state[1] += 1.0  # n10
        elif val < p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7:
            state[3] -= 1.0  # n11
            state[2] += 1.0  # n01
        elif val < p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8:
            state[4] += 1.0  # y (transcription)
        else:
            state[4] -= 1.0  # y (degradation)
            
    return state[4]


@njit
def fast_simulator_wrapper(alpha_s, beta_s, alpha_n, beta_n, k_y, gamma_y, t_max, num_cells):
    results = np.empty(num_cells)
    for i in range(num_cells):
        results[i] = fast_ssa_monomer(alpha_s, beta_s, alpha_n, beta_n, k_y, gamma_y, t_max)
        
    mean = np.mean(results)
    var = np.var(results)
    fano = var / mean if mean > 0 else 0.0
    cv = np.sqrt(var) / mean if mean > 0 else 0.0
    
    return np.log1p(np.array([mean, var, fano, cv]))

@njit
def fast_simulator_wrapper_het(alpha_s, beta_s, alpha_n, beta_n, k_y, gamma_y, t_max, num_cells):
    results = np.empty(num_cells)
    for i in range(num_cells):
        results[i] = fast_ssa_dimer(alpha_s, beta_s, alpha_n, beta_n, k_y, gamma_y, t_max)
        
    mean = np.mean(results)
    var = np.var(results)
    fano = var / mean if mean > 0 else 0.0
    cv = np.sqrt(var) / mean if mean > 0 else 0.0
    
    return np.log1p(np.array([mean, var, fano, cv]))


def pymc_fast_simulator(rng, alpha_s, beta_s, alpha_n, beta_n, k_y, gamma_y, t_max=None, num_cells=None, size=None):
    """
    Wrapper to bridge PyMC parameters with the Numba function.
    """
    a_s = np.asarray(alpha_s).item()
    b_s = np.asarray(beta_s).item()
    a_n = np.asarray(alpha_n).item()
    b_n = np.asarray(beta_n).item()
    g_y = np.asarray(gamma_y).item()
    k_y = np.asarray(k_y).item()

    # PyMC passes the desired shape via `size` based on `observed_data`.
    # Since observed_data is now 4 summary stats, we ignore `size` and 
    # simulate enough cells to get a good variance estimate (100 cells).
    n_cells = 100

    return fast_simulator_wrapper(
        a_s, 
        b_s, 
        a_n,
        b_n,
        k_y,
        g_y,
        t_max=1000, 
        num_cells=838
    )

def pymc_fast_simulator_het(rng, alpha_s, beta_s, alpha_n, beta_n, k_y, gamma_y, t_max=None, num_cells=None, size=None):
    """
    Wrapper to bridge PyMC parameters with the Numba function.
    """
    a_s = np.asarray(alpha_s).item()
    b_s = np.asarray(beta_s).item()
    a_n = np.asarray(alpha_n).item()
    b_n = np.asarray(beta_n).item()
    g_y = np.asarray(gamma_y).item()
    k_y = np.asarray(k_y).item()

    # Simulate 100 cells for robust variance estimate
    n_cells = 100

    return fast_simulator_wrapper_het(
        a_s, 
        b_s, 
        a_n,
        b_n,
        k_y,
        g_y,
        t_max=1000, 
        num_cells=838
    )
