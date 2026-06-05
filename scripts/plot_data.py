import os
import polars as pl
import pandas as pd

from expression_model.model_compile_data import _return_all_mfpt_trajectories
from expression_model.plotting_utils import (
    plot_mfpt_scatter,
    plot_fano_histogram,
    plot_mfpt_histogram,
)


def plot_individual_fanos(stats_df: pd.DataFrame, kbn_vals: list, kbs_vals: list, plot_dir: str):
    """Plot individual mRNA Fano factor histograms for each coordinate in the sweep."""
    os.makedirs(plot_dir, exist_ok=True)
    for kbn, kbs in zip(kbn_vals, kbs_vals):
        plot_fano_histogram(stats_df, kbn, kbs, plot_dir)


def plot_individual_mfpt_histograms(kbn_vals: list, kbs_vals: list, plot_dir: str, base_outdir: str):
    """Plot MFPT distributions for specific parameter coordinates."""
    os.makedirs(plot_dir, exist_ok=True)

    for kbn, kbs in zip(kbn_vals, kbs_vals):
        param_id = f"sweep/kbn_{kbn}_kbs_{kbs}"

        df_mfpt = _return_all_mfpt_trajectories(param_id=param_id, sim_time=1000, n_sites=10, output_dir=base_outdir)

        if isinstance(df_mfpt, pl.LazyFrame):
            df_mfpt = df_mfpt.collect()

        df_pandas = df_mfpt.to_pandas()

        # Filter for valid data
        target_col = "mature_lifespan_s"
        df_pandas = df_pandas.dropna(subset=[target_col])
        df_pandas = df_pandas[df_pandas[target_col] > 0]

        if df_pandas.empty:
            print(f"Skipping MFPT plot for {param_id}: No valid data.")
            continue

        output_path = f"{plot_dir}/mfpt_hist_kbn_{kbn}_kbs_{kbs}.png"
        plot_mfpt_histogram(
            df_pandas,
            output_path=output_path,
            title=f"MFPT Histogram (k_bind_n={kbn}, k_bind_s={kbs})",
        )


if __name__ == "__main__":
    stats_df = pl.read_parquet(snakemake.input.stats).to_pandas()
    kbn_vals_subset = snakemake.params.kbn_vals
    kbs_vals_subset = snakemake.params.kbs_vals
    base_outdir = snakemake.params.outdir

    plot_dir = os.path.dirname(snakemake.output.hue)

    # Plot 1: Scatter plot with MFPT hue
    plot_mfpt_scatter(stats_df, output_path=snakemake.output.hue, title="LHS Parameter Sweep: MFPT")

    # Plot 2: Individual Fano histograms
    plot_individual_fanos(stats_df, kbn_vals_subset, kbs_vals_subset, plot_dir)

    # Plot 3: Individual MFPT histograms
    plot_individual_mfpt_histograms(kbn_vals_subset, kbs_vals_subset, plot_dir, base_outdir)