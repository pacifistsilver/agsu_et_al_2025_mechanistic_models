import sys
import numpy as np
sys.path.append("src")

# snakemake --cores all --snakefile multiparam_set.smk  --config exp=src/config/hetmer_excl.yaml
configfile: config.get("exp", "src/config/default.yaml")
OUTDIR = config.get("output_dir", "heterodimer")

K_BIND_N_VALS_ALL = [f"{x:.2f}" for x in np.linspace(0.01, 1.0, num=100)]
K_BIND_S_VALS_ALL = [f"{x:.2f}" for x in np.linspace(0.01, 1.0, num=100)]

K_BIND_N_VALS_SUBSET = [f"{x:.2f}" for x in np.linspace(0.01, 1.0, num=10)]
K_BIND_S_VALS_SUBSET = [f"{x:.2f}" for x in np.linspace(0.01, 1.0, num=10)]

rule all:
    input:
        f"{OUTDIR}/plots/mean_residence_time_hue.png",
        expand(f"{OUTDIR}/plots/fano_hist_kbn_{{kbn}}_kbs_{{kbs}}.png", kbn=K_BIND_N_VALS_SUBSET, kbs=K_BIND_S_VALS_SUBSET),
        expand(f"{OUTDIR}/plots/mfpt_hist_kbn_{{kbn}}_kbs_{{kbs}}.png", kbn=K_BIND_N_VALS_SUBSET, kbs=K_BIND_S_VALS_SUBSET)
rule simulate:
    output:
        done=f"{OUTDIR}/sweep/kbn_{{kbn}}_kbs_{{kbs}}/simulation.done"
    params:
        kbn="{kbn}",
        kbs="{kbs}",
        outdir=OUTDIR,  # Pass the dynamic directory to the script
        config_path=config.get("exp", "src/config/default.yaml")
    threads: 12
    script:
        "scripts/run_sim_sweep.py"

rule compile_stats:
    input:
        expand(f"{OUTDIR}/sweep/kbn_{{kbn}}_kbs_{{kbs}}/simulation.done", kbn=K_BIND_N_VALS_ALL, kbs=K_BIND_S_VALS_ALL)
    output:
        f"{OUTDIR}/compiled_sweep_stats.parquet"
    script:
        "scripts/compile_data_sweep.py"

rule plot_results:
    input:
        stats=f"{OUTDIR}/compiled_sweep_stats.parquet",
        dones=expand(f"{OUTDIR}/sweep/kbn_{{kbn}}_kbs_{{kbs}}/simulation.done", kbn=K_BIND_N_VALS_ALL, kbs=K_BIND_S_VALS_ALL)
    output:
        hue=f"{OUTDIR}/plots/mean_residence_time_hue.png",
        # Snakemake will only expect the 200 subset images
        fano_hists=expand(f"{OUTDIR}/plots/fano_hist_kbn_{{kbn}}_kbs_{{kbs}}.png", kbn=K_BIND_N_VALS_SUBSET, kbs=K_BIND_S_VALS_SUBSET),
        mfpt_hists=expand(f"{OUTDIR}/plots/mfpt_hist_kbn_{{kbn}}_kbs_{{kbs}}.png", kbn=K_BIND_N_VALS_SUBSET, kbs=K_BIND_S_VALS_SUBSET)
    params:
        # Pass ONLY the subset lists to the plotting script!
        kbn_vals=K_BIND_N_VALS_SUBSET,
        kbs_vals=K_BIND_S_VALS_SUBSET,
        outdir=OUTDIR
    script:
        "scripts/plot_data.py"