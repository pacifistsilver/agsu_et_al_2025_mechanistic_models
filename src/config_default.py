from constants import SPECIES_NAMES, REACTION_NAMES, SPECIES_MAP, INITIAL_STATE_DEFAULTS, RATE_CONSTANTS_DEFAULTS

SPECIES_NAMES = SPECIES_NAMES
SPECIES_MAP = SPECIES_MAP
REACTION_NAMES = REACTION_NAMES

model_var = INITIAL_STATE_DEFAULTS
model_param = RATE_CONSTANTS_DEFAULTS

param_perturb_map = [
    ("k_unbind_s", 0),
]

# lhs
l_bounds = [0.1]
u_bounds = [1.0]
dimensions = 1
optimization = "random-cd"
samples = 100
out_dir = "output"

