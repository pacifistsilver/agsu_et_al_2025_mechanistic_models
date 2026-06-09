"""
Centralised configuration loading and merging utilities.

This module provides functions to load YAML config files and merge them with
default values, avoiding code duplication across scripts.
"""

import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional

import constants

def load_yaml_config(config_path: str) -> Dict[str, any]:
    """
    Load and parse a YAML configuration file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Parsed YAML as a dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


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
    # Start with defaults
    final_rates: Dict[str, float] = dict(constants.RATE_CONSTANTS_DEFAULTS)
    final_init_state: Dict[str, int] = dict(constants.INITIAL_STATE_DEFAULTS)

    # Override with config values
    final_rates.update(config.get("rates", {}))
    final_init_state.update(config.get("initial_state", {}))

    # Override with explicit custom values (highest priority)
    if custom_rates:
        final_rates.update(custom_rates)
    if custom_init_state:
        final_init_state.update(custom_init_state)

    return final_rates, final_init_state


def get_sim_parameters(config: Dict[str, any]) -> Dict[str, int | float]:
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


def get_output_dir(config: Dict[str, any]) -> str:
    """
    Get output directory from config or return default.

    Args:
        config: Loaded YAML configuration dict

    Returns:
        Output directory path
    """
    return config.get("output_dir", "output")


def get_experiment_name(config: Dict[str, any]) -> str:
    """
    Get experiment name from config or return default.

    Args:
        config: Loaded YAML configuration dict

    Returns:
        Experiment name string
    """
    return config.get("experiment_name", "default_experiment")
