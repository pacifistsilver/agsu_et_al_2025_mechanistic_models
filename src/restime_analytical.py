import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


#alpha_s = alpha_s
#alpha_n = alpha_n
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
def T_10():
    # Time to unbind starting from (Sox2 bound, Nanog unbound)
    factor1 = 1 / beta_s
    denominator = (beta_n * (1 - P_s1())) + (beta_s * (1 - P_n1()))
    return factor1 + (1 + (P_n1() / denominator))

def T_01(alpha_s, alpha_n):
    # Time to unbind starting from (Sox2 unbound, Nanog bound)
    factor1 = 1 / beta_n
    numerator = alpha_s * (alpha_n + beta_n)
    denominator = beta_s * (
        alpha_s + alpha_n + beta_s + beta_n
    )
    return factor1 * (1 + numerator / denominator)

def T_11():
    # Time to unbind starting from (Sox2 bound, Nanog bound)
    return (1 + beta_s * T_01() + beta_n * T_10()) / (
        beta_s + beta_n
    )

def T_bound():
    # Average residence time weighted by steady-state probabilities
    p_10 = P_s1() * (1 - P_n1())
    p_01 = (1 - P_s1()) * P_n1()
    p_11 = P_s1() * P_n1()

    numerator = p_10 * T_10() + p_01 * T_01() + p_11 * T_11()
    denominator = p_10 + p_01 + p_11
    return numerator / denominator

def Ts1():
    pn1 = P_n1()
    ps1 = P_s1()
    numerator = beta_s * pn1 * (beta_n + beta_s * (1 - pn1))
    denominator = beta_n * ((beta_n * (1 - ps1)) + beta_s * (1 - pn1))
    return (1 / beta_s) * (1 + (numerator / denominator))


sol = optimize.root_scalar(P_n1, bracket=[-1,1.0])
print(sol.root, sol.iterations, sol.function_calls)