import yaml
from src.expression_model.model_sample import run_single_parameter_set
from src.expression_model.config_default import model_param, model_var

if __name__ == "__main__":
    with open(snakemake.params.config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    rates = {**model_param, **cfg.get("rates", {})}
    states = {**model_var, **cfg.get("initial_state", {})}
    experiment_name = snakemake.params.experiment
    
    run_single_parameter_set(
        sim_max_time=cfg.get("sim_time", 1000),
        model_binding_sites=10,
        runs=100, 
        custom_initial_state=states,
        custom_rates=rates,
        param_set_id=experiment_name
    )

    # Touch the 'done' file so Snakemake knows this rule succeeded
    with open(snakemake.output[0], 'w') as f:
        f.write("Done.")