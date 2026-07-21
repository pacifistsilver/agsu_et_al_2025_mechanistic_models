import polars
import numpy as np
from abc_smc import fast_ssa_dimer

true_param = {
    "true_alpha_s": 0.5,
    "true_beta_s": 0.06,
    "true_alpha_n": 0.3,
    "true_beta_n": 0.2,
    "true_gamma_y": 0.005,
    "true_k_y": 0.1,
}

gene_count_vector = np.empty((40, 10))
for i in range(40):
    gene_count_vector[i] = fast_ssa_dimer(true_param["true_alpha_s"], 
                                    true_param["true_beta_s"],
                                    true_param["true_alpha_n"],
                                    true_param["true_beta_n"],
                                    true_param["true_k_y"],
                                    true_param["true_gamma_y"], 1000)

print(gene_count_vector.flatten())
np.save(file="./data/synthetic_data/synthetic_heterodimer_data", arr=gene_count_vector.flatten())