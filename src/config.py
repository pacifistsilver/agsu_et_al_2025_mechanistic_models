"""
Centralised configuration management, combining defaults and YAML loader.
"""

import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional

from . import constants

# Legacy config_default parameters for LHS/compilation
param_perturb_map = [
    ("k_unbind_s", 0),
]
l_bounds = [0.1]
u_bounds = [1.0]
dimensions = 1
optimization = "random-cd"
samples = 100
out_dir = "output"

model_var = constants.INITIAL_STATE_DEFAULTS
model_param = constants.RATE_CONSTANTS_DEFAULTS

def load_yaml_config(experiment_name: str, config_path: str = "src/experiments.yaml") -> Dict[str, any]:
    """
    Load and parse a specific experiment from the experiments YAML file.

    Args:
        experiment_name: Name of the experiment block to load.
        config_path: Path to the YAML config file

    Returns:
        Parsed YAML as a dictionary
    """
    with open(config_path, "r") as f:
        all_configs = yaml.safe_load(f) or {}
    
    if experiment_name not in all_configs:
        raise ValueError(f"Experiment '{experiment_name}' not found in {config_path}")
        
    return all_configs[experiment_name]


def merge_with_defaults(
    config: Dict[str, any],
    custom_rates: Optional[Dict[str, float]] = None,
    custom_init_state: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Merge user config with defaults, with custom overrides taking precedence.

    Args:
        config: Loaded YAML configuration dict
        custom_rates: Optional rate constants to override config
        custom_init_state: Optional initial state to override config

    Returns:
        Tuple of (final_rates, final_initial_state) dicts
    """
    final_rates: Dict[str, float] = dict(constants.RATE_CONSTANTS_DEFAULTS)
    final_init_state: Dict[str, int] = dict(constants.INITIAL_STATE_DEFAULTS)

    final_rates.update(config.get("rates", {}))
    final_init_state.update(config.get("initial_state", {}))

    if custom_rates:
        final_rates.update(custom_rates)
    if custom_init_state:
        final_init_state.update(custom_init_state)

    return final_rates, final_init_state


def get_sim_parameters(config: Dict[str, any]) -> Dict[str, any]:
    """
    Extract simulation parameters from config with sensible defaults.

    Args:
        config: Loaded YAML configuration dict

    Returns:
        Dict with 'sim_time', 'runs', 'binding_sites', 'burn_in_fraction'
    """
    return {
        "sim_time": config.get("sim_time", constants.MODEL_DEFAULTS["sim_time"]),
        "runs": config.get("runs", constants.MODEL_DEFAULTS["runs_per_parameter_set"]),
        "binding_sites": config.get("binding_sites", constants.MODEL_DEFAULTS["total_binding_sites"]),
        "burn_in_fraction": config.get("burn_in_fraction", constants.MODEL_DEFAULTS["burn_in_fraction"]),
    }
