import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_call import ModelCall
from src.config import model_param, model_var
import numpy as np


rates = model_param.copy()
rates['k_s_in'] = 10.0
rates['k_n_in'] = 5.0
rates['k_bind_s'] = 1.0
rates['k_bind_n'] = 1.0
rates['k_unbind_s'] = 0.5
rates['k_unbind_n'] = 0.5
rates['k_dimerise'] = 2.0
rates['k_dissociate'] = 1.0
rates['k_prod_m'] = 5.0
rates['k_deg_m'] = 0.1
rates['k_s_out'] = 0.5
rates['k_n_out'] = 0.5

print("Initializing model...")
model = ModelCall(
    model_param=rates,
    model_var=model_var,
    model_binding_sites=10,
    sim_max_time=100
)

print("Running trajectory...")
df_states, df_dwell, df_rxns = model.run_trajectory()

print("--- df_states ---")
print(df_states.head())

print("\n--- df_dwell ---")
print(df_dwell.head())

print("\n--- df_rxns ---")
print(df_rxns.head())

print("\nDone.")
