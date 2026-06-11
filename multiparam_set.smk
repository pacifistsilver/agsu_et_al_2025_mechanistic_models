import sys
import numpy as np
from scipy.stats import qmc

sys.path.append("src")

# snakemake --cores all -s multiparam_set.smk  --config exp=src/config/hetmer_excl.yaml

configfile: config.get("exp", "src/config/hetmer_excl.yaml")

NUM_SAMPLES = config.get("num_samples", 1)
SUBSET_SIZE = config.get("subset_size", 25)
OUTDIR = config.get("output_dir", "heterodimer")

sampler = qmc.LatinHypercube(d=2, seed=42)
sample = sampler.random(n=NUM_SAMPLES)
sample_scaled = qmc.scale(sample, [0.01, 0.01], [1.0, 1.0])

K_BIND_N_VALS_ALL = [f"{val[0]:.2f}" for val in sample_scaled]
K_BIND_S_VALS_ALL = [f"{val[1]:.2f}" for val in sample_scaled]

K_BIND_N_VALS_SUBSET = K_BIND_N_VALS_ALL[:SUBSET_SIZE]
K_BIND_S_VALS_SUBSET = K_BIND_S_VALS_ALL[:SUBSET_SIZE]

rule all:
    input:
        f"{OUTDIR}/plots/mean_residence_time_hue.png"

rule simulate:
    output:
        sweep_dir=directory(f"{OUTDIR}/sweep/kbn_{{kbn}}_kbs_{{kbs}}")
    params:
        kbn="{kbn}",
        kbs="{kbs}",
        outdir=OUTDIR,  
        config_path=config.get("exp", "src/config/hetmer_excl.yaml")
    threads: 8
    log:
        f"{OUTDIR}/logs/simulate/kbn_{{kbn}}_kbs_{{kbs}}.log"
    script:
        "scripts/run_sim_sweep.py"

rule compile_stats:
    input:
        sweep_dirs=expand(f"{OUTDIR}/sweep/kbn_{{kbn}}_kbs_{{kbs}}", zip, kbn=K_BIND_N_VALS_ALL, kbs=K_BIND_S_VALS_ALL)
    output:
        f"{OUTDIR}/compiled_sweep_stats.parquet"
    log:
        f"{OUTDIR}/logs/compile_stats.log"
    script:
        "scripts/compile_data_sweep.py"

rule plot_results:
    input:
        stats=f"{OUTDIR}/compiled_sweep_stats.parquet",
        sweep_dirs=expand(f"{OUTDIR}/sweep/kbn_{{kbn}}_kbs_{{kbs}}", zip, kbn=K_BIND_N_VALS_ALL, kbs=K_BIND_S_VALS_ALL)
    output:
        hue=f"{OUTDIR}/plots/mean_residence_time_hue.png"
    params:
        outdir=OUTDIR
    log:
        f"{OUTDIR}/logs/plot_results.log"
    script:
        "scripts/plot_data.py"