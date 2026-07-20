import polars
import numpy as np
from abc_smc import pymc_fast_simulator

true_param = {
    "true_alpha_s": 0.5,
    "true_beta_s": 0.06,
    "true_alpha_n": 0.3,
    "true_beta_n": 0.2,
    "true_gamma_y": 0.005,
    "true_k_y": 0.01,
}
observed_data = pymc_fast_simulator(None, true_param["true_alpha_s"], 
                                    true_param["true_beta_s"],
                                    true_param["true_alpha_n"],
                                    true_param["true_beta_n"],
                                    true_param["true_k_y"],
                                    true_param["true_gamma_y"], 1000, 40, None)

print(observed_data)
np.save(file="sim_data", arr=observed_data)