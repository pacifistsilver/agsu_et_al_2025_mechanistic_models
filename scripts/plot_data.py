import os
import polars as pl
from src.plotting_utils import plot_mfpt_scatter

if __name__ == "__main__":
    stats_df = pl.read_parquet(snakemake.input.stats).to_pandas()
    base_outdir = snakemake.params.outdir

    # Plot 1: Scatter plot with MFPT hue
    plot_mfpt_scatter(stats_df, output_path=snakemake.output.hue, title="LHS Parameter Sweep: MFPT")