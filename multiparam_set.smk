import sys
import numpy as np
from scipy.stats import qmc

sys.path.append("src")

# snakemake --cores all --snakefile multiparam_set.smk  --config exp=src/config/hetmer_excl.yaml

NUM_SAMPLES = 1000
sampler = qmc.LatinHypercube(d=2)
sample = sampler.random(n=NUM_SAMPLES)
sample_scaled = qmc.scale(sample, [0.01, 0.01], [1.0, 1.0])

configfile: config.get("exp", "src/config/default.yaml")
OUTDIR = config.get("output_dir", "heterodimer")

K_BIND_N_VALS_ALL = [f"{val[0]:.2f}" for val in sample_scaled]
K_BIND_S_VALS_ALL = [f"{val[1]:.2f}" for val in sample_scaled]

SUBSET_SIZE = 25  # Generate Fano/MFPT histograms for the first 25 samples
K_BIND_N_VALS_SUBSET = K_BIND_N_VALS_ALL[:SUBSET_SIZE]
K_BIND_S_VALS_SUBSET = K_BIND_S_VALS_ALL[:SUBSET_SIZE]

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
        outdir=OUTDIR,  
        config_path=config.get("exp", "src/config/default.yaml")
    script:
        "scripts/run_sim_sweep.py"

rule compile_stats:
    input:
        expand(f"{OUTDIR}/sweep/kbn_{{kbn}}_kbs_{{kbs}}/simulation.done", zip, kbn=K_BIND_N_VALS_ALL, kbs=K_BIND_S_VALS_ALL)
    output:
        f"{OUTDIR}/compiled_sweep_stats.parquet"
    script:
        "scripts/compile_data_sweep.py"

rule plot_results:
    input:
        stats=f"{OUTDIR}/compiled_sweep_stats.parquet",
        dones=expand(f"{OUTDIR}/sweep/kbn_{{kbn}}_kbs_{{kbs}}/simulation.done", zip, kbn=K_BIND_N_VALS_ALL, kbs=K_BIND_S_VALS_ALL)
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