"""
Expression model package for stochastic Gillespie simulations of gene expression
with spatial chromatin dynamics.

Key modules:
- model: Core Gillespie simulator and state management
- model_sample: Simulation execution and parameter sampling
- model_stats: Statistical feature extraction from trajectories
- model_compile_data: Data aggregation and compilation
- config_loader: Configuration loading and merging
- constants: Centralised constants and configuration values
- plotting_utils: Shared plotting utilities and visualization functions
"""

from . import constants
from .model import ModelCall, ModelState, TranscriptionFactor, ModelLogger, ModelReactions
from .model_sample import (
    run_single_parameter_set,
    execute_simulation,
    run_and_save_trajectory,
    generate_lhs_and_run,
)
from .model_stats import Statistics
from .model_compile_data import compile_all_runs, _return_all_mfpt_trajectories, parse_parameters_txt
from .config_loader import (
    load_yaml_config,
    merge_with_defaults,
    get_sim_parameters,
    get_output_dir,
    get_experiment_name,
)
from .plotting_utils import (
    plot_mfpt_scatter,
    plot_mfpt_contour,
    plot_fano_histogram,
    plot_mfpt_histogram,
    plot_fano_comparison,
    plot_mfpt_comparison,
)

__all__ = [
    "constants",
    "ModelCall",
    "ModelState",
    "TranscriptionFactor",
    "ModelLogger",
    "ModelReactions",
    "run_single_parameter_set",
    "execute_simulation",
    "run_and_save_trajectory",
    "generate_lhs_and_run",
    "Statistics",
    "compile_all_runs",
    "_return_all_mfpt_trajectories",
    "parse_parameters_txt",
    "load_yaml_config",
    "merge_with_defaults",
    "get_sim_parameters",
    "get_output_dir",
    "get_experiment_name",
    "plot_mfpt_scatter",
    "plot_mfpt_contour",
    "plot_fano_histogram",
    "plot_mfpt_histogram",
    "plot_mfpt_vs_parameter",
    "plot_fano_comparison",
    "plot_mfpt_comparison",
]
