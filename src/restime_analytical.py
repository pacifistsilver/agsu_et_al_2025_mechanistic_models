import numpy as np
import matplotlib.pyplot as plt


class HeterodimerModel:
    def __init__(self, alpha_s, beta_s, alpha_n, beta_n):
        self.alpha_s = alpha_s
        self.beta_s = beta_s
        self.alpha_n = alpha_n
        self.beta_n = beta_n

    # --- Probabilities ---
    def P_s1(self):
        return self.alpha_s / (self.alpha_s + self.beta_s)

    def P_n1(self):
        return self.alpha_n / (self.alpha_n + self.beta_n)

    def P_bound(self):
        # P_bound = 1 - P(s=0)*P(n=0)
        denominator = (self.alpha_s + self.beta_s) * (self.alpha_n + self.beta_n)
        numerator = self.beta_s * self.beta_n 
        return 1 - (numerator / denominator)

    # --- Mean First-Passage Times (Residence Times) ---
    def T_10(self):
        # Time to unbind starting from (Sox2 bound, Nanog unbound)
        factor1 = 1 / self.beta_s
        denominator = (self.beta_n * (1 - self.P_s1())) + (self.beta_s * (1 - self.P_n1()))
        return factor1 + (1 + (self.P_n1() / denominator))

    def T_01(self):
        # Time to unbind starting from (Sox2 unbound, Nanog bound)
        factor1 = 1 / self.beta_n
        numerator = self.alpha_s * (self.alpha_n + self.beta_n)
        denominator = self.beta_s * (
            self.alpha_s + self.alpha_n + self.beta_s + self.beta_n
        )
        return factor1 * (1 + numerator / denominator)

    def T_11(self):
        # Time to unbind starting from (Sox2 bound, Nanog bound)
        return (1 + self.beta_s * self.T_01() + self.beta_n * self.T_10()) / (
            self.beta_s + self.beta_n
        )

    def T_bound(self):
        # Average residence time weighted by steady-state probabilities
        p_10 = self.P_s1() * (1 - self.P_n1())
        p_01 = (1 - self.P_s1()) * self.P_n1()
        p_11 = self.P_s1() * self.P_n1()

        numerator = p_10 * self.T_10() + p_01 * self.T_01() + p_11 * self.T_11()
        denominator = p_10 + p_01 + p_11
        return numerator / denominator

    def Ts1(self):
        pn1 = self.P_n1()
        ps1 = self.P_s1()
        numerator = self.beta_s * pn1 * (self.beta_n + self.beta_s * (1 - pn1))
        denominator = self.beta_n * ((self.beta_n * (1 - ps1)) + self.beta_s * (1 - pn1))
        return (1 / self.beta_s) * (1 + (numerator / denominator))


