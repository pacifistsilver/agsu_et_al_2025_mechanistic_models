import sys
import os
sys.path.insert(0, os.path.abspath("."))

from src.config_loader import load_yaml_config, merge_with_defaults, get_sim_parameters
from src.sample_model import run_single_parameter_set

if __name__ == "__main__":
    kbn = float(snakemake.params.kbn)
    kbs = float(snakemake.params.kbs)

    base_outdir = snakemake.params.outdir
    param_set_id = f"sweep/kbn_{kbn:.2f}_kbs_{kbs:.2f}"

    cfg = load_yaml_config(snakemake.params.config_path)

    final_rates, final_state = merge_with_defaults(
        cfg,
        custom_rates={"k_bind_n": kbn, "k_bind_s": kbs},
    )

    sim_params = get_sim_parameters(cfg)

    run_single_parameter_set(
        sim_max_time=sim_params["sim_time"],
        runs=sim_params["runs"],
        model_binding_sites=sim_params["binding_sites"],
        custom_rates=final_rates,
        custom_initial_state=final_state,
        param_set_id=param_set_id,
        output_dir=base_outdir,
        max_workers=snakemake.threads,
        activator_tf=cfg.get("activator_tf", "sox2")
    )