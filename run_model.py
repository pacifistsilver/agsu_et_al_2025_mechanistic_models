import os
import polars as pl
import concurrent.futures
import numpy as np
from scipy.stats import qmc 
from plot_trajectories import ModelPlot
from main import ModelCall

os.makedirs("output/states", exist_ok=True)
os.makedirs("output/dwell_times", exist_ok=True)
os.makedirs("output/reactions", exist_ok=True)

default_model_var = {
    "sox2_monomer_free": 5, 
    "nanog_monomer_free": 10, 
    "sox2_monomer_bound": 0, 
    "nanog_monomer_bound": 0, 
    "nanog_sox2_dimer_bound": 0, 
    "nanog_nanog_dimer_bound": 0,
    "nanog_sox2_dimer_free": 0,
    "nanog_nanog_dimer_free": 0,
    "nanog_sox2_dimer_single_bound": 0,
    "nanog_nanog_dimer_single_bound": 0,
    "mRNA": 0
}

default_model_param = {
    "k_s_in": 0, "k_s_out": 0, 
    "k_n_in": 0, "k_in_out": 0,
    "k_bind_s": 1.0, "k_unbind_s": 0.06,
    "k_bind_n": 1.0, "k_unbind_n": 0.25,
    "k_dimerise": 1.0,  
    "k_prod_m": 1.0,    
    "k_deg_m": 0.53, 
    "k_dissociate": 0.5
}

def run_and_save_trajectory(run_id: int, sample_params: dict):
    """Wrapper to run the model, tag the data, and save to Parquet."""
    
    # Initialize your model with the sampled parameters
    model = ModelCall(
        model_param=sample_params["initial_state"], 
        model_var=sample_params["rates"], 
        model_binding_sites=10, 
        sim_max_time=5000, 
        track_history=False
    )
    
    df_states, df_dwell, df_rxns = model.run_trajectory()
    
    mrna_counts = df_states["mRNA"].to_numpy()
    mrna_mean = mrna_counts.mean()
    mrna_var = mrna_counts.var()
    mrna_covariance = np.sqrt(mrna_var) / mrna_mean if mrna_mean > 0 else 0
    mrna_fano = mrna_var / mrna_mean
    
    df_states = df_states.with_columns(pl.lit(run_id).alias("run_id"))
    df_dwell = df_dwell.with_columns(pl.lit(run_id).alias("run_id"))
    df_rxns = df_rxns.with_columns(pl.lit(run_id).alias("run_id"))
    df_states.write_parquet(f"output/states/run_{run_id}.parquet")
    df_dwell.write_parquet(f"output/dwell_times/run_{run_id}.parquet")
    df_rxns.write_parquet(f"output/reactions/run_{run_id}.parquet")
    
    
    sox2_dwells = df_dwell.filter(pl.col("species") == "SOX2")["dwell_time"]
    nanog_dwells = df_dwell.filter(pl.col("species") == "NANOG")["dwell_time"]
    sox2_mean_dwell = sox2_dwells.mean() if len(sox2_dwells) > 0 else 0.0
    nanog_mean_dwell= nanog_dwells.mean() if len(nanog_dwells) > 0 else 0.0
    return {
        "run_id": run_id,
        "mrna_mean": mrna_mean,
        "mrna_fano": mrna_fano,
        "mrna_covariance": mrna_covariance,
        "sox2_mean_dwell": sox2_mean_dwell,
        "nanog_mean_dwell": nanog_mean_dwell
    }

def generate_lhs_and_run(num_samples: int):
    sampler = qmc.LatinHypercube(d=6, optimization="random-cd")
    sample = sampler.random(n=num_samples)
    # kbinds, kunbinds, kbindn, kunbindn, k_dimerise, kdissociate,
    l_bounds = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    u_bounds = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)    
    metadata = []
    for i in range(num_samples):
        metadata.append({
            "run_id": i,
            "k_bind_s": sample_scaled[i][0],
            "k_unbind_s": sample_scaled[i][1],
            "k_bind_n": sample_scaled[i][2],
            "k_unbind_n": sample_scaled[i][3],
            "k_dimerise": sample_scaled[i][4],
            "k_dissociate": sample_scaled[i][5],
        })
    
    pl.DataFrame(metadata).write_parquet("output/lhs_metadata.parquet")
    df_meta = pl.DataFrame(metadata)
    summary_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for meta in metadata:
            run_rates = default_model_param.copy() 
            run_rates["k_bind_s"] = meta["k_bind_s"]
            run_rates["k_unbind_s"] = meta["k_unbind_s"]
            run_rates["k_bind_n"] = meta["k_bind_n"]
            run_rates["k_unbind_n"] = meta["k_unbind_n"]
            run_rates["k_dimerise"] = meta["k_dimerise"]
            run_rates["k_dissociate"] = meta["k_dissociate"]
            model_var = default_model_var.copy()
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
            save_path = "output/lhs_summary_results.parquet"
            df_final.write_parquet(save_path)
            print(f"Successfully saved {len(df_final)} runs to {save_path}")
        
if __name__ == '__main__': 
    generate_lhs_and_run(num_samples=200)
