import yaml
from expression_model.config_default import model_param, model_var
from expression_model.model_sample import run_single_parameter_set

if __name__ == "__main__":
    kbn = float(snakemake.params.kbn)
    kbs = float(snakemake.params.kbs)
    
    # Grab the custom output directory
    base_outdir = snakemake.params.outdir
    param_set_id = f"sweep/kbn_{kbn:.2f}_kbs_{kbs:.2f}"

    with open(snakemake.params.config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    final_rates = {**model_param, **cfg.get("rates", {})}
    final_rates["k_bind_n"] = kbn
    final_rates["k_bind_s"] = kbs
    final_state = {**model_var, **cfg.get("initial_state", {})}

    # Pass the outdir into the output parameter!
    run_single_parameter_set(
        sim_max_time=cfg.get("sim_time", 1000),
        runs=cfg.get("runs", 100),
        model_binding_sites=10,
        custom_rates=final_rates,
        custom_initial_state=final_state,
        param_set_id=param_set_id,
        output_dir=base_outdir   
    )

    with open(snakemake.output.done, 'w') as f:
        f.write("Done.")