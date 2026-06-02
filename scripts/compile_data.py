from model_compile_data import _return_all_mfpt_trajectories

experiment_name = snakemake.params.experiment
folder_prefix = f"param_set_{experiment_name}"

# Run your existing compiler function
_return_all_mfpt_trajectories(
    param_id=folder_prefix, 
    sim_time=1000, 
    n_sites=10
)