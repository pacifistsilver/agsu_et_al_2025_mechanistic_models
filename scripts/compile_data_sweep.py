import os
import json
import polars as pl
from eda_functions import Statistics

if __name__ == "__main__":
    all_run_features = []

    print(f"Compiling {len(snakemake.input)} parameter configurations...")

    # 1. Iterate through all the simulation.done files Snakemake tracked
    for done_file in snakemake.input:
        # The folder is the directory containing the .done file
        folder = os.path.dirname(done_file)
        param_set_id = os.path.basename(folder)
        
        # Load the parameters saved by the simulation script
        params_path = os.path.join(folder, "model_parameters.json")
        with open(params_path, "r") as f:
            params_data = json.load(f)
            
        rates = params_data["rates"]
        initial_state = params_data["initial_state"]

        # Extract features using your Statistics class
        stats = Statistics.from_parquet_dir(
            param_set_dir=folder, 
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