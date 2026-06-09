"""Helper script to call main.py

Obtains many realisations of the stochastic simulation, saving this data per simulation run to a .parquet file.
This file is stored in the ./output folder created during the simulation

Typical usage may look like on a CLI:
    python run_model.py

If you face any issues, email: dwl25@ic.ac.uk or danielluo1143@gmail.com
"""

import os
import json
import polars as pl
import concurrent.futures
import numpy as np
import config_default as config
from scipy.stats import qmc
from .model_call import ModelCall
from .model_utils import TranscriptionFactor


output = config.out_dir
os.makedirs(output, exist_ok=True)


def run_and_save_trajectory(
    run_id: int,
    param_set_id: int,
    sample_params: dict,
    param_set_dir: str,
    sim_max_time: int,
    model_binding_sites: int,
):
    """Wrapper to run the model, tag the data, and save to Parquet."""

    SOX2 = TranscriptionFactor(id=1, name="SOX2", valency=2, is_activator=True)
    NANOG = TranscriptionFactor(id=2, name="NANOG", valency=2, is_activator=False)
    
    model = ModelCall(
        tfs=[SOX2, NANOG],
        model_param=sample_params["rates"],
        model_var=sample_params["initial_state"],
        model_binding_sites=model_binding_sites,
        sim_max_time=sim_max_time,
    )

    df_states, df_dwell, df_rxns = model.run_trajectory()

    # Tag the data with BOTH the parameter set ID and the specific run ID
    df_states = df_states.with_columns(
        [pl.lit(param_set_id).alias("param_set_id"), pl.lit(run_id).alias("run_id")]
    )
    df_dwell = df_dwell.with_columns(
        [pl.lit(param_set_id).alias("param_set_id"), pl.lit(run_id).alias("run_id")]
    )
    df_rxns = df_rxns.with_columns(
        [pl.lit(param_set_id).alias("param_set_id"), pl.lit(run_id).alias("run_id")]
    )

    # Save to their respective folders
    df_states.write_parquet(
        os.path.join(param_set_dir, "states", f"run_{run_id}.parquet")
    )
    df_dwell.write_parquet(
        os.path.join(param_set_dir, "dwell_times", f"run_{run_id}.parquet")
    )
    df_rxns.write_parquet(os.path.join(param_set_dir, "rxns", f"run_{run_id}.parquet"))

    return {
        "param_set_id": param_set_id,
        "run_id": run_id,
    }


def generate_lhs_and_run(
    num_samples: int,
    dimensions: int,
    runs_per_param_set: int = 100,
    optimization: str = "random-cd",
):
    sampler = qmc.LatinHypercube(d=dimensions, optimization=optimization)
    sample = sampler.random(n=num_samples)
    sample_scaled = qmc.scale(sample, config.l_bounds, config.u_bounds)

    metadata = []

    print(f"Generating LHS parameters for {num_samples} parameter sets...")

    for i in range(num_samples):
        submission = {"param_set_id": i}
        for param in config.param_perturb_map:
            submission[param[0]] = sample_scaled[i][param[1]]
        metadata.append(submission)

    df_meta = pl.DataFrame(metadata)
    summary_results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = []

        for meta in metadata:
            param_set_id = meta["param_set_id"]

            run_rates = config.model_param.copy()
            model_var = config.model_var.copy()

            for param_name, param_val in meta.items():
                if param_name == "param_set_id":
                    continue
                if param_name in run_rates:
                    run_rates[param_name] = param_val
                else:
                    model_var[param_name] = param_val

            run_params = {"initial_state": model_var, "rates": run_rates}

            param_set_dir = os.path.join(output, str(param_set_id))
            os.makedirs(os.path.join(param_set_dir, "states"), exist_ok=True)
            os.makedirs(os.path.join(param_set_dir, "dwell_times"), exist_ok=True)
            os.makedirs(os.path.join(param_set_dir, "rxns"), exist_ok=True)

            with open(os.path.join(param_set_dir, "model_parameters.txt"), "w") as f:
                f.write("--- Simulation Parameters ---\n")
                f.write(json.dumps(run_rates, indent=4))
                f.write("\n\n--- Initial State ---\n")
                f.write(json.dumps(model_var, indent=4))
                f.write(f"\n\nTotal Runs Set: {runs_per_param_set}")

            # Submit 100 runs for this specific parameter set
            for run_id in range(runs_per_param_set):
                futures.append(
                    executor.submit(
                        run_and_save_trajectory,
                        run_id,
                        param_set_id,
                        run_params,
                        param_set_dir,
                    )
                )

        total_jobs = num_samples * runs_per_param_set
        completed = 0
        print(
            f"Submitted {total_jobs} total simulations ({num_samples} sets x {runs_per_param_set} runs). Executing..."
        )

        for future in concurrent.futures.as_completed(futures):
            try:
                completed_run_data = future.result()
                summary_results.append(completed_run_data)
                completed += 1
                if completed % 100 == 0:
                    print(f"Completed {completed}/{total_jobs} runs...")
            except Exception as e:
                print(f"A simulation run failed: {e}")

        if summary_results:
            print("\nAggregating LHS summary results and saving to Parquet...")
            df_summary = pl.DataFrame(summary_results)

            df_final = df_meta.join(df_summary, on="param_set_id")
            save_path = os.path.join(output, "lhs_summary_results.parquet")
            df_final.write_parquet(save_path)
            print(f"Successfully saved aggregated data to {save_path}")

def run_single_parameter_set(
    sim_max_time: int,
    model_binding_sites: int,
    runs: int = 100,
    param_set_id: str = "",
    custom_rates: dict = None,
    custom_initial_state: dict = None,
    output_dir: str = "output"  
):
    """
    Runs multiple stochastic trajectories for a given parameter/variable set.
    """
    run_rates = config.model_param.copy()
    if custom_rates:
        run_rates.update(custom_rates)

    model_var = config.model_var.copy()
    if custom_initial_state:
        model_var.update(custom_initial_state)

    run_params = {"initial_state": model_var, "rates": run_rates}

    meta = {"param_set_id": param_set_id}
    meta.update(run_rates)
    meta.update(model_var)
    df_meta = pl.DataFrame([meta])
    
    param_set_dir = os.path.join(output_dir, str(param_set_id))
    os.makedirs(os.path.join(param_set_dir, "states"), exist_ok=True)
    os.makedirs(os.path.join(param_set_dir, "dwell_times"), exist_ok=True)
    os.makedirs(os.path.join(param_set_dir, "rxns"), exist_ok=True)

    with open(os.path.join(param_set_dir, "model_parameters.txt"), "w") as f:
        f.write("--- Simulation Parameters ---\n")
        f.write(json.dumps(run_rates, indent=4))
        f.write("\n\n--- Initial State ---\n")
        f.write(json.dumps(model_var, indent=4))
        f.write(f"\n\nTotal Runs Set: {runs}")

    with open(os.path.join(param_set_dir, "model_parameters.json"), "w") as f:
        json.dump({"rates": run_rates, "initial_state": model_var}, f, indent=4)

    summary_results = []

    print(f"Submitting {runs} baseline simulations for parameter set {param_set_id}...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for run_id in range(runs):
            futures.append(
                executor.submit(
                    run_and_save_trajectory,
                    run_id,
                    param_set_id,
                    run_params,
                    param_set_dir,
                    sim_max_time,
                    model_binding_sites,
                )
            )

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                completed_run_data = future.result()
                summary_results.append(completed_run_data)
                completed += 1

                if completed % max(1, runs // 10) == 0 or completed == runs:
                    print(f"Completed {completed}/{runs} runs...")
            except Exception as e:
                print(f"A simulation run failed: {e}")

    if summary_results:
        print("\nAggregating single parameter set summary results...")
        df_summary = pl.DataFrame(summary_results)

        df_final = df_meta.join(df_summary, on="param_set_id")

        save_path = os.path.join(output_dir, f"{param_set_id}_summary_results.parquet")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True) 
        
        df_final.write_parquet(save_path)
        print(f"Successfully saved aggregated data to {save_path}")      

def execute_simulation(run_id: int, param_set: dict) -> dict:
    """Purely runs the math and returns raw data in memory."""
    
    # Define biology
    SOX2 = TranscriptionFactor(id=1, name="SOX2", valency=2, is_activator=True)
    NANOG = TranscriptionFactor(id=2, name="NANOG", valency=2, is_activator=False)
    
    model = ModelCall(
        tfs=[SOX2, NANOG], # Add biology
        model_param=param_set["rates"],
        model_var=param_set["initial_state"],
        model_binding_sites=10, # Add missing arg
        sim_max_time=1000       # Add missing arg
    )
    df_states, df_dwell, df_rxns = model.run_trajectory()
    
    return {
        "run_id": run_id,
        "states": df_states,
        "dwell": df_dwell,
        "rxns": df_rxns
    }

def save_simulation_data(sim_data: dict, output_dir: str):
    """Purely handles file I/O."""
    run_id = sim_data["run_id"]
    
    # Write to parquet
    sim_data["states"].write_parquet(f"{output_dir}/states/run_{run_id}.parquet")
    sim_data["dwell"].write_parquet(f"{output_dir}/dwell_times/run_{run_id}.parquet")
    # ...

