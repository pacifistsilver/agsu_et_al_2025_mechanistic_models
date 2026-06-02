import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

experiment_name = snakemake.params.experiment
input_parquet = snakemake.input[0]
output_png = snakemake.output[0]

# Determine settings based on experiment name
require_multistep = False if "MONOMER" in experiment_name else True

# Your plotting logic
df = pl.read_parquet(input_parquet).to_pandas()
target_col = "mature_lifespan_s" if require_multistep else "transient_lifespan_s"

df = df.dropna(subset=[target_col])
df = df[df[target_col] > 0]
df = df[~df["starting_species"].isin(["SOX2b:NANOGf", "NANOGb:SOX2f"])]

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(
    data=df, x=target_col, hue="starting_species",
    element="step", stat="density", common_norm=False,    
    alpha=0.3, linewidth=2,
)

plt.title(f"{experiment_name} MFPT Histogram", fontsize=14)
plt.xlabel("MFPT (s)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.tight_layout()

# Save the plot precisely where Snakemake expects it
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.close()