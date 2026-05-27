import os
import glob
import json
import polars as pl
from model_stats import Statistics
import model_config as config

def parse_parameters_txt(filepath):
    """Safely extracts the dictionaries from the saved text file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    parts = content.split('--- Initial State ---')
    rates_str = parts[0].replace('--- Simulation Parameters ---\n', '').strip()
    states_str = parts[1].split('Total Runs Set:')[0].strip()
    
    return json.loads(rates_str), json.loads(states_str)

def compile_all_runs():
    base_dir = config.out_dir
    param_folders = glob.glob(os.path.join(base_dir, "param_set_0"))
    
    all_run_features = []
    
    print(f"Found {len(param_folders)} parameter sets. Extracting features per run...")
    
    for folder in param_folders:
        try:
            # Extract ID from folder name (e.g., param_set_12 -> 12)
            param_set_id = int(os.path.basename(folder).split("_")[-1])
            
            # Read specific parameters for this LHS sample
            params_path = os.path.join(folder, "model_parameters.txt")
            rates, initial_state = parse_parameters_txt(params_path)
            
            # Load the ensemble lazily
            stats = Statistics.from_parquet_dir(
                param_set_dir=folder, 
                total_sim_time=1000, # Make sure this matches max_time in your LHS script!
                total_binding_sites=10
            )
            
            # Extract features grouped by run_id
            df_features = stats.extract_per_run_features(param_set_id, rates, initial_state)
            all_run_features.append(df_features)
            
            print(f"Processed param_set_{param_set_id} ({len(df_features)} runs)")
            
        except Exception as e:
            print(f"Skipping {folder} due to error: {e}")
            
    if not all_run_features:
        print("No data extracted. Exiting.")
        return
        
    print("\nConcatenating master dataset...")
    
    # Use diagonal concatenation in case some parameter sets generated dimer types that others didn't
    master_df = pl.concat(all_run_features, how="diagonal").fill_null(0.0)
    
    # Save to disk
    csv_out = os.path.join(base_dir, "master_per_run_features.csv")
    parquet_out = os.path.join(base_dir, "master_per_run_features.parquet")
    
    master_df.write_csv(csv_out)
    master_df.write_parquet(parquet_out)
    
    print(f"\nSUCCESS! Master dataset saved to:")
    print(f" -> {csv_out}")
    print(f" -> {parquet_out}")
    print(f"Total Rows (Individual Runs): {len(master_df)}")
    print(f"Total Columns (Features): {len(master_df.columns)}")

if __name__ == "__main__":
    compile_all_runs()