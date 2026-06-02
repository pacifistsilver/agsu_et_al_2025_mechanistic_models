import os
import src.features.model_config as config
from src.features.model_sample import run_single_parameter_set


if __name__ == "__main__":
    experiment_name = snakemake.params.experiment

    run_rates = config.model_param.copy()
    model_var = config.model_var.copy()

    run_single_parameter_set(
        sim_max_time=1000,
        model_binding_sites=10,
        runs=100, 
        custom_initial_state=model_var,
        custom_rates=run_rates,
        param_set_id=experiment_name
    )

    # Touch the 'done' file so Snakemake knows this rule succeeded
    with open(snakemake.output[0], 'w') as f:
        f.write("Done.")