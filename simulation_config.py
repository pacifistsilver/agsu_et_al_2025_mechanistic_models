model_var = {
    "sox2_monomer_free": 10, 
    "nanog_monomer_free": 1, 
    "sox2_monomer_bound": 0, 
    "nanog_monomer_bound": 0, 
    "nanog_sox2_dimer_bound": 0, 
    "nanog_nanog_dimer_bound": 0,
    "nanog_sox2_dimer_free": 0,
    "nanog_nanog_dimer_free": 0,
    "nanog_sox2_dimer_single_bound": 0,
    "nanog_nanog_dimer_single_bound": 0,
    "mRNA": 0
}

model_param = {
    "k_s_in": 0, "k_s_out": 0,
    "k_n_in": 0, "k_in_out": 0, 
    "k_bind_s": 1.0, "k_unbind_s": 0.06,
    "k_bind_n": 1.0, "k_unbind_n": 0.24,
    "k_dimerise": 0.0,  
    "k_prod_m": 1.0,    
    "k_deg_m": 0.53, 
    "k_dissociate": 0
}

param_perturb_map = [
    ("k_bind_s", 0),
    ("k_unbind_s", 1),
    ("k_bind_n", 2),
    ("k_unbind_n", 3),    
]

l_bounds = [0.1, 0.1, 0.1, 0.1]
u_bounds = [1.0, 1.0, 1.0, 1.0]
dimensions = 4
optimization = "random-cd"
samples = 5
out_dir = "output"

