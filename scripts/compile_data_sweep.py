import sys
import os
sys.path.insert(0, os.path.abspath("."))

import json
import polars as pl
from src.eda_functions import Statistics

if __name__ == "__main__":
    sweep_dirs = snakemake.input.sweep_dirs
    all_run_features = []

    print(f"Compiling {len(sweep_dirs)} parameter configurations...")

    # 1. Iterate through all the simulation folders
    for sweep_dir in sweep_dirs:
        param_set_id = os.path.basename(sweep_dir)
        
        # Load the parameters saved by the simulation script
        params_path = os.path.join(sweep_dir, "model_parameters.json")
        with open(params_path, "r") as f:
            params_data = json.load(f)
            
        rates = params_data["rates"]
        initial_state = params_data["initial_state"]

        # Extract features using your Statistics class
        stats = Statistics.from_parquet_dir(
            param_set_dir=sweep_dir, 
            total_sim_time=1000, 
            total_binding_sites=10
        )
        
        df_features = stats.extract_per_run_features(param_set_id, rates, initial_state)
        all_run_features.append(df_features)
        
        print(f"Processed {param_set_id}")

    # 2. Concatenate everything and save exactly where Snakemake expects it
    if all_run_features:
        print("\nConcatenating master sweep dataset...")
        master_df = pl.concat(all_run_features, how="diagonal").fill_null(0.0)
        
        # Save to snakemake.output[0], which is "output/compiled_sweep_stats.parquet"
        master_df.write_parquet(snakemake.output[0])
        print(f"SUCCESS! Master dataset saved to {snakemake.output[0]}")
    else:
        raise FileNotFoundError("No data extracted. Compilation failed.")