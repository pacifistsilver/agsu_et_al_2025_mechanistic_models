import sys
sys.path.append("src")
from expression_model.config_default import model_param, model_var

configfile: config.get("exp", "src/config/default.yaml")

rule all:
    input:
        expand("output/param_set_{exp}_all_mfpt_histogram_data.parquet", 
               exp=config["experiment_name"])        
# sim
rule simulate:
    output:
        "output/param_set_{experiment}/simulation.done"
    params:
        config_path = config.get("exp", "src/config/default.yaml")    ,
        experiment = config["experiment_name"]
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

