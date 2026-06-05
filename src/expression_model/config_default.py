# Import all constants from centralized constants module
from . import constants

# For backward compatibility, expose constants and defaults at module level
SPECIES_NAMES = constants.SPECIES_NAMES
SPECIES_MAP = constants.SPECIES_MAP
REACTION_NAMES = constants.REACTION_NAMES

model_var = constants.INITIAL_STATE_DEFAULTS
model_param = constants.RATE_CONSTANTS_DEFAULTS

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

