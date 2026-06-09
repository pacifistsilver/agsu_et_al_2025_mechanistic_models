"""
Constants for species names, reactions, data visualisation, and model parameters.

This module serves as the single source of truth for all constants used across
the project.
"""

SPECIES_NAMES = [
    "sox2_monomer_free",
    "nanog_monomer_free",
    "sox2_monomer_bound",
    "nanog_monomer_bound",
    "nanog_sox2_dimer_bound",
    "nanog_nanog_dimer_bound",
    "nanog_sox2_dimer_free",
    "nanog_nanog_dimer_free",
    "nanog_sox2_dimer_single_bound",
    "nanog_nanog_dimer_single_bound",
    "mRNA",
]

SPECIES_MAP = {name: i for i, name in enumerate(SPECIES_NAMES)}

DISPLAY_SPECIES_NAMES = {
    "sox2_monomer_bound": "SOX2b",
    "nanog_monomer_bound": "NANOGb",
    "nanog_sox2_dimer_free": "NANOGf:SOX2f",
    "nanog_nanog_dimer_free": "NANOGf:NANOGf",
    "nanog_sox2_dimer_bound": "NANOGb:SOX2b",
    "nanog_sox2_dimer_single_bound": "NANOGb:SOX2f",
    "nanog_nanog_dimer_bound": "NANOGb:NANOGb",
    "nanog_nanog_dimer_single_bound": "NANOGb:NANOGf",
}


REACTION_NAMES = {
    0: "prod_s",
    1: "prod_n",
    2: "bind_s",
    3: "bind_n",
    4: "deg_s",
    5: "deg_n",
    6: "unbind_s",
    7: "unbind_n",
    8: "prod_m",
    9: "deg_m",
    10: "site_dimerise",
    11: "bulk_dimerise",
    12: "site_dedimerise",
    13: "bulk_dedimerise",
    14: "site_bulk_dimerise",
    15: "single_dimer_dedimerise",
}

### Data Visualisation

SPECIES_PALETTE = {
    "SOX2b": "#1f77b4",
    "NANOGb": "#ff7f0e",
    "NANOGb:SOX2f": "#8CC0EB",
    "SOX2b:NANOGb": "#d62728",
    "NANOGb:NANOGf": "#9467bd",
    "SOX2b:NANOGf": "#3A5B7C",
    "NANOGb:NANOGb": "#e377c2",
    "Heterodimer": "#DAA464",
}

### Data Processing
COLUMN_PREFIXES = {
    "mfpt": "MFPT_",  # Mean first passage time
    "duration": "mean_duration_",  # Mean dwell time
    "param": "param_",  # Parameter columns (e.g., param_k_bind_n)
    "init": "init_",  # Initial state columns
    "fano": "mrna_fano",  # Fano factor for mRNA
}

SWEEP_PARAM_COLS = {
    "k_bind_n": "param_k_bind_n",
    "k_bind_s": "param_k_bind_s",
}

### Model Parameter Defaults

MODEL_DEFAULTS = {
    "total_binding_sites": 10,
    "sim_time": 1000,  # seconds
    "runs_per_parameter_set": 100,
    "burn_in_fraction": 0.2,  # 20% of sim_time for mRNA stats
    "lhs_samples": 100,
}

RATE_CONSTANTS_DEFAULTS = {
    "k_s_in": 0,
    "k_s_out": 0,
    "k_n_in": 0,
    "k_in_out": 0,
    "k_bind_s": 0.0,
    "k_unbind_s": 0.0,
    "k_bind_n": 0.0,
    "k_unbind_n": 0.0,
    "k_dimerise": 0,
    "k_prod_m": 0.0,
    "k_deg_m": 0.0,
    "k_dissociate": 0,
}

INITIAL_STATE_DEFAULTS = {
    "sox2_monomer_free": 0,
    "nanog_monomer_free": 0,
    "sox2_monomer_bound": 0,
    "nanog_monomer_bound": 0,
    "nanog_sox2_dimer_bound": 0,
    "nanog_nanog_dimer_bound": 0,
    "nanog_sox2_dimer_free": 0,
    "nanog_nanog_dimer_free": 0,
    "nanog_sox2_dimer_single_bound": 0,
    "nanog_nanog_dimer_single_bound": 0,
    "mRNA": 0,
}


### LHS Config

LHS_CONFIG = {
    "dimensions": 2,
    "samples": 100,
    "seed": 42,
    "optimization": "random-cd",
    "bounds": {
        "k_bind_n": {"lower": 0.01, "upper": 1.0},
        "k_bind_s": {"lower": 0.01, "upper": 1.0},
    },
}

### TF Data
TRANSCRIPTION_FACTORS = {
    "sox2": {
        "id": 1,
        "full_name": "SOX2",
        "abbreviation": "S",
        "is_activator": True,
    },
    "nanog": {
        "id": 2,
        "full_name": "NANOG",
        "abbreviation": "N",
        "is_activator": False,
    },
}