import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# model odes
def sox2_system(state, t, kts, kdegs, kcs, kucs, kc, kuc, n, beta, gamma, thetac):
    d_free, d_cluster, s2_free, s2_bound, m_count = state
    d_cluster_safe = max(0, d_cluster)
    s2_bound_safe = max(0, s2_bound)
        
    # Hill function for SOX2 binding (driven by clustered DNA)
    hill_sox2_bind = (d_cluster_safe / thetac)**n / (1 + (d_cluster_safe / thetac)**n)
    # Hill function for DNA clustering (driven by bound SOX2)
    hill_cluster = (s2_bound_safe / thetac)**n / (1 + (s2_bound_safe / thetac)**n)
    
    # 2. DNA Structural Fractions
    d_dfree = kuc * d_cluster - (kc * hill_cluster * d_free)
    d_dcluster = (kc * hill_cluster * d_free) - kuc * d_cluster
    
    # 3. SOX2 Trafficking
    d_sfree = kts - (kcs * hill_sox2_bind * s2_free) + (kucs * s2_bound) - (kdegs * s2_free)
    d_sbound = (kcs * hill_sox2_bind * s2_free) - (kucs * s2_bound)
    
    # 4. Transcriptional Output
    d_m = (beta * d_sbound) - (gamma * m_count)
    
    return [d_dfree, d_dcluster, d_sfree, d_sbound, d_m]


def cdx2_system(state, t, kts, ktc, kdegs, kdegc, kcs, kucs, kbc, kubc, kl, kul, n, beta, gamma, thetac):
    # Fixed state unpacking to perfectly match the return order
    d_unlooped, d_looped, c_free, s2_free, s2_bound, m_count = state
    
    # Hill function for SOX2 binding (driven by looped DNA)
    hill_sox2_bind = (d_looped / thetac)**n / (1 + (d_looped / thetac)**n)
    
    # 2. DNA Structural Fractions 
    d_dunlooped = kul * d_looped - kl * d_unlooped - kbc * c_free * d_unlooped + kubc * d_looped
    d_dlooped = kl * d_unlooped + kbc * c_free * d_unlooped - kul * d_looped - kubc * d_looped
    
    # 3. CDX2 Trafficking
    d_cfree = ktc + kul * d_looped + kubc * d_looped - kdegc * c_free - kbc * c_free * d_unlooped
    
    # 4. SOX2 Trafficking
    d_sfree = kts - kcs * hill_sox2_bind * s2_free + kucs * s2_bound - kdegs * s2_free
    d_sbound = kcs * hill_sox2_bind * s2_free - kucs * s2_bound
    
    # 5. Transcriptional Output
    d_m = beta * s2_bound - gamma * m_count

    return [d_dunlooped, d_dlooped, d_cfree, d_sfree, d_sbound, d_m]

# state variables
ode_state_variables = {
    "case_sox2" : {
        "m_count": 0.0,  
        "s2_free": 100,       
        "s2_bound": 0,   
        "d_free": 100,    
        "d_cluster": 0
    },
    "case_cdx2": {
        "m_count": 0.0,  
        "s2_free": 0.0,       
        "s2_bound": 0.0,   
        "d_free": 0.0,    
        "d_looped": 0.0,
        "c_free": 0.0
    }
}

# model parameters
ode_parameters = {
    "case_sox2": {
        "kt_s": 1, 
        "kdeg_s": 0.05,
        "kconfine_s": 1.0, 
        "kunconfine_s": 0.1, 
        "kcluster": 1.0,
        "kuncluster": 0.01,
        "beta": 1,
        "gamma": 0.03,
        "theta_c": 10, # 
        "n": 2
    }, 
    "case_cdx2": {
        "kt_s": 1, 
        "kt_c": 1,
        "kdeg_c": 0.05,
        "kdeg_s": 0.05,
        "kconfine_s": 0.05, 
        "kunconfine_s": 0.02, 
        "kbind_c": 0.04, 
        "kunbind_c": 0.02, 
        "kloop": 0.05,
        "kunloop": 0.02,
        "beta": 1,
        "gamma": 0.03,
        "theta_c": 1,
        "n": 2
    }
}

# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0244219#sec002
# Constant for mRNA synthesis/degradation: αim = 1/ min βim = 0.03/ min
# Constant for protein synthesis/degradation: αip = 1/ min βip = 0.2/ min

## todo: plot, check equations are correct,

t = np.linspace(0, 100, 1000)

y0_sox2 = [
    ode_state_variables["case_sox2"]["d_free"],
    ode_state_variables["case_sox2"]["d_cluster"],
    ode_state_variables["case_sox2"]["s2_free"],
    ode_state_variables["case_sox2"]["s2_bound"],
    ode_state_variables["case_sox2"]["m_count"]
]
params_sox2 = (
    ode_parameters["case_sox2"]["kt_s"],
    ode_parameters["case_sox2"]["kdeg_s"],
    ode_parameters["case_sox2"]["kconfine_s"],
    ode_parameters["case_sox2"]["kunconfine_s"],
    ode_parameters["case_sox2"]["kcluster"],
    ode_parameters["case_sox2"]["kuncluster"],
    ode_parameters["case_sox2"]["n"],
    ode_parameters["case_sox2"]["beta"],
    ode_parameters["case_sox2"]["gamma"],
    ode_parameters["case_sox2"]["theta_c"]
)
solution_sox2 = odeint(sox2_system, y0_sox2, t, args=params_sox2)
print(solution_sox2)
plt.figure(figsize=(10, 6))

plt.plot(t, solution_sox2[:, 0], label='Free DNA', color='red',linestyle='--')
plt.plot(t, solution_sox2[:, 1], label='Clustered DNA', color = 'black', linestyle='--')
plt.plot(t, solution_sox2[:, 2], label='Free SOX2', linewidth=2)
plt.plot(t, solution_sox2[:, 3], label='Bound SOX2', linewidth=2)
plt.plot(t, solution_sox2[:, 4], label='mRNA', color='purple', linewidth=2)

plt.title('SOX2 System')
plt.xlabel('Time (t)')
plt.ylabel('Concentration')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()