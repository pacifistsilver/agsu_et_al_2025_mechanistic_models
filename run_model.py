"""Helper script to call main.py

Obtains many realisations of the stochastic simulation, saving this data per simulation run to a .parquet file. 
This file is stored in the ./output folder created during the simulation

Typical usage may look like on a CLI:
    python run_model.py

If you face any issues, email: dwl25@ic.ac.uk or danielluo1143@gmail.com 
"""
import os
import polars as pl
import concurrent.futures
import numpy as np
import simulation_config as config
from scipy.stats import qmc 
from plot_trajectories import ModelPlot
from model import ModelCall

output = config.out_dir

os.makedirs(f"{output}", exist_ok=True)

def run_and_save_trajectory(run_id: int, sample_params: dict):
    """Wrapper to run the model, tag the data, and save to Parquet."""
    
    model = ModelCall(
        model_param=sample_params["initial_state"], 
        model_var=sample_params["rates"], 
        model_binding_sites=10, 
        sim_max_time=5000, 
        track_history=False
    )
    
    df_states, df_dwell, df_rxns = model.run_trajectory()    
    df_states = df_states.with_columns(pl.lit(run_id).alias("run_id"))
    df_dwell = df_dwell.with_columns(pl.lit(run_id).alias("run_id"))
    df_rxns = df_rxns.with_columns(pl.lit(run_id).alias("run_id"))

    mrna_counts = df_states["mRNA"].to_numpy()
    sox2_dwells = df_dwell.filter(pl.col("species") == "SOX2")["dwell_time"]
    nanog_dwells = df_dwell.filter(pl.col("species") == "NANOG")["dwell_time"]
    sox2_dwell_counts = sox2_dwells.to_numpy()    
    nanog_dwell_counts = nanog_dwells.to_numpy()

    return {
        "run_id": run_id,
        "model_var": sample_params["rates"],
        "model_initial_state": sample_params["initial_state"],
        "mrna_counts": mrna_counts,
        "sox2_dwell_counts": sox2_dwell_counts,
        "nanog_dwell_counts": nanog_dwell_counts,    
    }

def generate_lhs_and_run(num_samples: int, dimensions:int, optimization:str = "random-cd"):
    sampler = qmc.LatinHypercube(d=dimensions, optimization=optimization)
    sample = sampler.random(n=num_samples)
    # kbinds, kbindn, k_dimerise, kdissociate, sox2 free, nanog free
    sample_scaled = qmc.scale(sample, config.l_bounds, config.u_bounds)    
    metadata = []
    # vary the number of parameters we are passing to metadata dependent on X
    # number of parameters to pass to metadata dependent on length of model.param_perturb_map
    
    for i in range(num_samples):
        submission = {}
        submission = submission | {
            "run_id": i,
        }
        for param in config.param_perturb_map:  
            perturbed_param_dict = {
                param[0]: sample_scaled[i][param[1]]
            }
            submission = submission | perturbed_param_dict
        metadata.append(submission)

    df_meta = pl.DataFrame(metadata)
    summary_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for meta in metadata:
            run_rates = config.model_param.copy() 
            model_var = config.model_var.copy()
            for rate in meta: 
                run_rates[rate] = meta[rate]
            run_params = {"initial_state": model_var, "rates": run_rates}
            
            futures.append(executor.submit(run_and_save_trajectory, meta["run_id"], run_params))
            
        for future in concurrent.futures.as_completed(futures):
            completed_run_data = future.result()
            summary_results.append(completed_run_data)
            print(f"Completed run {completed_run_data['run_id']}")
        
        if summary_results:
            print("\nAggregating results and saving to Parquet...")
            df_summary = pl.DataFrame(summary_results)
            df_final = df_meta.join(df_summary, on="run_id")
            save_path = f"{output}/lhs_summary_results.parquet"
            df_final.write_parquet(save_path)
            print(f"Successfully saved {len(df_final)} run(s) to {save_path}")
        
if __name__ == '__main__': 
    generate_lhs_and_run(num_samples=config.samples, dimensions=config.dimensions, optimization=config.optimization)
