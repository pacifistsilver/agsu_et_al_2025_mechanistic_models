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

SPECIES_NAMES = [
    "sox2_monomer_free", "nanog_monomer_free", "sox2_monomer_bound", "nanog_monomer_bound",
    "nanog_sox2_dimer_bound", "nanog_nanog_dimer_bound", "nanog_sox2_dimer_free", 
    "nanog_nanog_dimer_free", "nanog_sox2_dimer_single_bound", "nanog_nanog_dimer_single_bound", "mRNA"
]
SPECIES_MAP = {name: i for i, name in enumerate(SPECIES_NAMES)}

REACTION_NAMES = {
    0: "prod_s", 1: "prod_n", 2: "bind_s", 3: "bind_n", 4: "deg_s", 5: "deg_n", 
    6: "unbind_s", 7: "unbind_n", 8: "prod_m", 9: "deg_m", 10: "site_dimerise", 
    11: "bulk_dimerise", 12: "tether_bind", 13: "site_dedimerise", 14: "bulk_dedimerise", 15: "site_bulk_dimerise"
}


param_perturb_map = [
    ("k_bind_s", 0),
    ("k_unbind_s", 1),
    ("k_bind_n", 2),
    ("k_unbind_n", 3),    
]


# lhs
l_bounds = [0.1, 0.1, 0.1, 0.1]
u_bounds = [1.0, 1.0, 1.0, 1.0]
dimensions = 4
optimization = "random-cd"
samples = 5
out_dir = "output"

