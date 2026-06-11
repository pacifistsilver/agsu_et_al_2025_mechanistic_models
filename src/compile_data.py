import os
import glob
import json
import polars as pl
from . import config_default as config
from .eda_functions import Statistics



def parse_parameters_txt(filepath):
    """Extracts the dictionaries from the saved text file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    parts = content.split('--- Initial State ---')
    rates_str = parts[0].replace('--- Simulation Parameters ---\n', '').strip()
    states_str = parts[1].split('Total Runs Set:')[0].strip()
    
    return json.loads(rates_str), json.loads(states_str)

def compile_all_runs(param_id: str):
    base_dir = config.out_dir
    param_folders = glob.glob(os.path.join(base_dir, param_id))
    
    all_run_features = []
    
    print(f"Found {len(param_folders)} parameter sets. Extracting features per run...")
    
    for folder in param_folders:
        try:
            # Extract ID from folder name (e.g., param_set_12 -> 12)
            param_set_id = os.path.basename(folder).replace("param_set_", "")
            
            params_path = os.path.join(folder, "model_parameters.txt")
            rates, initial_state = parse_parameters_txt(params_path)
            
            stats = Statistics.from_parquet_dir(
                param_set_dir=folder, 
                total_sim_time=1000,
                total_binding_sites=10
            )
            
            df_features = stats.extract_per_run_features(param_set_id, rates, initial_state)
            all_run_features.append(df_features)
            
            print(f"Processed param_set_{param_set_id} ({len(df_features)} runs)")
            
        except Exception as e:
            print(f"Skipping {folder} due to error: {e}")
            
    if not all_run_features:
        raise FileNotFoundError(f"No data extracted for {param_id}. Cannot create output parquet.")
        return
        
    print("\nConcatenating master dataset...")
    
    master_df = pl.concat(all_run_features, how="diagonal").fill_null(0.0)
    
    csv_out = os.path.join(base_dir, f"{param_id}_stats.csv")
    parquet_out = os.path.join(base_dir, f"{param_id}_stats.parquet")
    
    master_df.write_csv(csv_out)
    master_df.write_parquet(parquet_out)
    
    print(f"\nSUCCESS! Master dataset saved to:")
    print(f" -> {csv_out}")
    print(f" -> {parquet_out}")
    print(f"Total Rows (Individual Runs): {len(master_df)}")
    print(f"Total Columns (Features): {len(master_df.columns)}")

def _return_all_mfpt_trajectories(
    param_id: str, 
    sim_time: int, 
    n_sites: int = 10,
    output_dir: str = "output"   
):
    """Compiles MFPT data from a specific parameter set directory into a single dataframe."""
    print(f"Extracting all MFPT trajectories for {param_id}...")
    
    param_dir = os.path.join(output_dir, str(param_id))
    
    if not os.path.exists(param_dir):
        raise FileNotFoundError(f"Parameter directory does not exist: {param_dir}")

    try:
        stats = Statistics.from_parquet_dir(
            param_set_dir=param_dir, 
            total_sim_time=sim_time, 
            total_binding_sites=n_sites
        )
        
        df_features = stats._return_all_trajectories()
        
        if isinstance(df_features, pl.LazyFrame):
            df_features = df_features.collect()
            
    except Exception as e:
        print(f"Error extracting trajectories from {param_dir}: {e}")
        raise e
        
    if df_features.is_empty():
        print(f"No trajectory data found for {param_id}.")
        return df_features
        
    param_basename = os.path.basename(param_dir)
    parquet_out = os.path.join(param_dir, f"{param_basename}_all_mfpt_histogram_data.parquet")
    
    df_features.write_parquet(parquet_out)
    print(f"SUCCESS! Trajectories saved to: {parquet_out}")
    
    return df_features

def _return_binding_frequencies(
    param_id: str, 
    sim_time: int, 
    n_sites: int = 10,
    output_dir: str = "output"   
):
    """Compiles binding frequency data per site from a specific parameter set directory."""
    print(f"Extracting binding frequencies for {param_id}...")
    
    param_dir = os.path.join(output_dir, str(param_id))
    
    if not os.path.exists(param_dir):
        raise FileNotFoundError(f"Parameter directory does not exist: {param_dir}")

    try:
        stats = Statistics.from_parquet_dir(
            param_set_dir=param_dir, 
            total_sim_time=sim_time, 
            total_binding_sites=n_sites
        )
        
        df_freq = stats._calc_binding_frequencies_per_site()
        
        if isinstance(df_freq, pl.LazyFrame):
            df_freq = df_freq.collect()
            
    except Exception as e:
        print(f"Error extracting binding frequencies from {param_dir}: {e}")
        raise e
        
    param_basename = os.path.basename(param_dir)
    parquet_out = os.path.join(param_dir, f"{param_basename}_binding_frequencies.parquet")
    
    df_freq.write_parquet(parquet_out)
    print(f"SUCCESS! Binding frequencies saved to: {parquet_out}")
    
    return df_freq.to_pandas()