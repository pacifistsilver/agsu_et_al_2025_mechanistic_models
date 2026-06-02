EXPERIMENTS = ["nanog_heterodimer_excl", "nanog_dimer_mixed"]

rule all:
    input:
        expand("output/param_set_{experiment}_all_mfpt_histogram_data.parquet", experiment=EXPERIMENTS)
        
# sim
rule simulate:
    output:
        "output/param_set_{experiment}/simulation.done"
    params:
        experiment = "{experiment}"
    script:
        "scripts/run_simulation.py"

# compile
rule compile:
    input:
        "output/param_set_{experiment}/simulation.done"
    output:
        "output/param_set_{experiment}_all_mfpt_histogram_data.parquet"
    params:
        experiment = "{experiment}"
    script:
        "scripts/compile_data.py"

