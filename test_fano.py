import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class HeterodimerBindingModel:
    """
    Calculates steady-state probabilities and Mean First-Passage Times (MFPT) 
    for a heterodimer DNA binding model (e.g., Sox2 and Nanog).
    """
    def __init__(self, alpha_s, beta_s, alpha_n, beta_n):
        self.alpha_s = alpha_s  # Binding rate of Sox2
        self.beta_s = beta_s    # Unbinding rate of Sox2
        self.alpha_n = alpha_n  # Binding rate of Nanog
        self.beta_n = beta_n    # Unbinding rate of Nanog

    # --- State Probabilities and Steady State ---

    def p_s1(self):
        """Probability of Sox2 monomer bound to DNA (Eq. 2)"""
        return self.alpha_s / (self.alpha_s + self.beta_s)

    def p_n1(self):
        """Probability of Nanog monomer bound to DNA (Eq. 3)"""
        return self.alpha_n / (self.alpha_n + self.beta_n)

    def p_bound(self):
        """Probability of finding the heterodimer bound to DNA (Eq. 7)"""
        p_s0 = self.beta_s / (self.alpha_s + self.beta_s)
        p_n0 = self.beta_n / (self.alpha_n + self.beta_n)
        return 1 - (p_s0 * p_n0)

    # --- Duration of the Bound State (MFPT) ---

    def t_10(self):
        """
        MFPT from singly bound state (1,0) to fully unbound (0,0) (Eq. 14).
        Represents lifetime of a heterodimer where only Sox2 is initially bound.
        """
        term1 = 1 / self.beta_s
        numerator = self.alpha_n * (self.alpha_s + self.beta_s)
        denominator = self.beta_n * (self.alpha_s + self.alpha_n + self.beta_s + self.beta_n)
        return term1 * (1 + (numerator / denominator))

    def t_bound(self):
        """
        Average MFPT weighted by the steady-state probability of occupying 
        each bound state (Eq. 16).
        """
        p_s1 = self.p_s1()
        p_n1 = self.p_n1()
        p_bound = self.p_bound()
        
        num = (self.beta_n**2 * p_s1 * (1 - p_s1)) + \
              (self.beta_s**2 * p_n1 * (1 - p_n1)) + \
              (self.beta_s * self.beta_n * p_bound)
        
        den = (self.beta_s * self.beta_n * p_bound) * \
              (self.beta_n * (1 - p_s1) + self.beta_s * (1 - p_n1))
        
        return num / den

    def t_s1(self):
        """
        MFPT assuming an observed bound heterodimer is dominated by the 
        Sox2 bound state (Eq. 17).
        """
        p_s1 = self.p_s1()
        p_n1 = self.p_n1()
        
        term1 = 1 / self.beta_s
        num = self.beta_s * p_n1 * (self.beta_n + self.beta_s * (1 - p_n1))
        den = self.beta_n * (self.beta_n * (1 - p_s1) + self.beta_s * (1 - p_n1))
        
        return term1 * (1 + (num / den))


# ==========================================
# Parameter Sweep and Visualization
# ==========================================
if __name__ == "__main__":
    # Fixed Sox2 rates
    alpha_s = 0.5
    beta_s = 0.06  # Monomer baseline lifetime is 1/beta_s = 1.0
    
    # Sweep Nanog binding rates to see effect on heterodimer persistence
    alpha_n_vals = np.linspace(0.1, 1.0, 100)
    beta_n = 0.2
    
    results = []
    
    for a_n in alpha_n_vals:
        model = HeterodimerBindingModel(alpha_s=alpha_s, beta_s=beta_s, alpha_n=a_n, beta_n=beta_n)
        
        results.append({
            'Nanog_Binding_Rate_Alpha_N': a_n,
            'P_Sox2_Bound': model.p_s1(),
            'P_Heterodimer_Bound': model.p_bound(),
            'T10_Residence_Time': model.t_10(),
            'T_Bound_Average': model.t_bound(),
            'T_S1_Dominated': model.t_s1()
        })
        
    df_results = pd.DataFrame(results)

    # Visualization of the extended binding time
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the baseline monomer lifetime
    ax.axhline(1 / beta_s, color='black', linestyle='--', label='Sox2 Monomer Lifetime ($1/\\beta_s$)')

    # Plot the heterodimer residence times
    sns.lineplot(data=df_results, x='Nanog_Binding_Rate_Alpha_N', y='T10_Residence_Time', 
                 label='$T_{10}$ (Initially Sox2 bound)', linewidth=2.5, ax=ax)
    sns.lineplot(data=df_results, x='Nanog_Binding_Rate_Alpha_N', y='T_Bound_Average', 
                 label='$T_{bound}$ (Steady-state average)', linewidth=2.5, ax=ax)

    ax.set_title('Heterodimer Effective Binding Time vs. Nanog Binding Rate', fontsize=14)
    ax.set_xlabel('Nanog Binding Rate ($\\alpha_n$)', fontsize=12)
    ax.set_ylabel('Mean First-Passage Time (MFPT)', fontsize=12)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()