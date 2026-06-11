import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

beta_s = 0.06
beta_n = 0.24

def P_s1(alpha_s):
    return alpha_s / (alpha_s + beta_s)

def P_s0(alpha_s):
    return beta_s / (beta_s + alpha_s)

def P_n1(alpha_n):
    return alpha_n / (alpha_n + beta_n)

def P_n0(alpha_n):
    return beta_n / (beta_n + alpha_n)

def P_bound(alpha_s, alpha_n):
    # P_bound = 1 - P(s=0)*P(n=0)
    denominator = (alpha_s + beta_s) * (alpha_n + beta_n)
    numerator = beta_s * beta_n 
    return 1 - (numerator / denominator)

# --- Mean First-Passage Times (Residence Times) ---
def T_10(alpha_s, alpha_n):
    # Time to unbind starting from (Sox2 bound, Nanog unbound)
    factor1 = 1 / beta_s
    denominator = (beta_n * (1 - P_s1(alpha_s))) + (beta_s * (1 - P_n1(alpha_n)))
    return factor1 + (1 + (P_n1(alpha_n) / denominator))

def T_01(alpha_s, alpha_n):
    # Time to unbind starting from (Sox2 unbound, Nanog bound)
    factor1 = 1 / beta_n
    numerator = alpha_s * (alpha_n + beta_n)
    denominator = beta_s * (
        alpha_s + alpha_n + beta_s + beta_n
    )
    return factor1 * (1 + numerator / denominator)

def T_11(alpha_s, alpha_n):
    # Time to unbind starting from (Sox2 bound, Nanog bound)
    return (1 + beta_s * T_01(alpha_s, alpha_n) + beta_n * T_10(alpha_s, alpha_n)) / (
        beta_s + beta_n
    )

def T_bound(alpha_s, alpha_n):
    # Average residence time weighted by steady-state probabilities
    p_10 = P_s1(alpha_s) * (1 - P_n1(alpha_n))
    p_01 = (1 - P_s1(alpha_s)) * P_n1(alpha_n)
    p_11 = P_s1(alpha_s) * P_n1(alpha_n)

    numerator = p_10 * T_10(alpha_s, alpha_n) + p_01 * T_01(alpha_s, alpha_n) + p_11 * T_11(alpha_s, alpha_n)
    denominator = p_10 + p_01 + p_11
    return numerator / denominator

def Ts1(alpha_s, alpha_n):
    pn1 = P_n1(alpha_n)
    ps1 = P_s1(alpha_s)
    numerator = beta_s * pn1 * (beta_n + beta_s * (1 - pn1))
    denominator = beta_n * ((beta_n * (1 - ps1)) + beta_s * (1 - pn1))
    return (1 / beta_s) * (1 + (numerator / denominator))

def Tn1(alpha_s, alpha_n):
    # Symmetric effective residence time for NANOG
    pn1 = P_n1(alpha_n)
    ps1 = P_s1(alpha_s)
    numerator = beta_n * ps1 * (beta_s + beta_n * (1 - ps1))
    denominator = beta_s * ((beta_s * (1 - pn1)) + beta_n * (1 - ps1))
    return (1 / beta_n) * (1 + (numerator / denominator))

def target_equations(alphas, target_Ts, target_Tn):
    alpha_s, alpha_n = alphas
    return [
        Ts1(alpha_s, alpha_n) - target_Ts,
        Tn1(alpha_s, alpha_n) - target_Tn
    ]

# assuming some target effective residence times (e.g. 17.0 for SOX2, 25.0 for NANOG)
target_Ts = 23
target_Tn = 25
initial_guess = [0.1, 0.1]

sol = optimize.root(target_equations, initial_guess, args=(target_Ts, target_Tn))

print(f"Target Ts: {target_Ts}s, Target Tn: {target_Tn}s")
if sol.success:
    alpha_s_opt, alpha_n_opt = sol.x
    print(f"Converged!")
    print(f"Calculated alpha_s: {alpha_s_opt:.4f}")
    print(f"Calculated alpha_n: {alpha_n_opt:.4f}")
    print(f"Verification - Ts1: {Ts1(alpha_s_opt, alpha_n_opt):.4f}, Tn1: {Tn1(alpha_s_opt, alpha_n_opt):.4f}")
else:
    print(f"Root finder failed to converge: {sol.message}")
