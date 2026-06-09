"""
Shared plotting utilities for expression model.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import constants
from typing import Optional
from src.logger import setup_logger

logger = setup_logger(__name__)


def setup_plot(figsize: tuple = (8, 6), style: str = "whitegrid") -> tuple:
    """
    Initialise plot with standard theme and figure size.

    Args:
        figsize: (width, height) in inches
        style: Seaborn style theme

    Returns:
        Tuple of (fig, ax)
    """
    sns.set_theme(style=style)
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def get_species_palette() -> dict:
    """Get the standard species color palette."""
    return constants.SPECIES_PALETTE.copy()

def get_target_column(
    df: pd.DataFrame,
    preferred: str = "MFPT_SOX2b:NANOGf",
    column_prefix: str = "MFPT_",
) -> Optional[str]:
    """
    Find target column for plotting, with preference fallback.

    Tries to find a preferred column first, then falls back to first available
    column matching the prefix.

    Args:
        df: DataFrame to search
        preferred: Preferred column name
        column_prefix: Prefix to match for fallback search

    Returns:
        Column name if found, None otherwise
    """
    if preferred in df.columns:
        return preferred

    matching_cols = [c for c in df.columns if c.startswith(column_prefix)]
    return matching_cols[0] if matching_cols else None


def ensure_output_dir(output_path: str):
    """Create output directory if it doesn't exist."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def plot_mfpt_scatter(
    df: pd.DataFrame,
    x_col: str = "param_k_bind_n",
    y_col: str = "param_k_bind_s",
    hue_col: Optional[str] = None,
    output_path: Optional[str] = None,
    title: str = "MFPT Landscape",
    figsize: tuple = (8, 6),
) -> tuple:
    """
    Create scatter plot of parameters with MFPT as hue.

    Args:
        df: DataFrame with parameter and MFPT columns
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        hue_col: Column name for color hue (auto-detected if None)
        output_path: Path to save figure, or None to not save
        title: Plot title
        figsize: Figure size

    Returns:
        Tuple of (fig, ax)
    """
    fig, ax = setup_plot(figsize=figsize)

    if hue_col is None:
        hue_col = get_target_column(df)

    if hue_col is None:
        # No MFPT data, plot plain scatter
        sns.scatterplot(data=df, x=x_col, y=y_col, color="grey", s=200, ax=ax)
        ax.set_title(title + " (No binding events)")
    else:
        sns.scatterplot(
            data=df,
            x=x_col,
            y=y_col,
            hue=hue_col,
            palette="viridis",
            s=200,
            edgecolor="k",
            ax=ax,
        )
        species_name = hue_col.replace("MFPT_", "")
        ax.set_title(f"{title} ({species_name})")
        ax.legend(title="Mean Residence Time (s)", bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_xlabel(x_col.replace("param_", ""))
    ax.set_ylabel(y_col.replace("param_", ""))
    plt.tight_layout()

    if output_path:
        ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=300)
        plt.close()
        return None, None

    return fig, ax


def plot_fano_histogram(
    df: pd.DataFrame,
    kbn: str,
    kbs: str,
    output_dir: str,
    figsize: tuple = (8, 5),
) -> None:
    """
    Plot mRNA Fano factor histogram for a specific parameter coordinate.

    Args:
        df: Full statistics DataFrame
        kbn: k_bind_n parameter value
        kbs: k_bind_s parameter value
        output_dir: Directory to save plot
        figsize: Figure size
    """
    # Filter for specific coordinate
    subset_df = df[
        (df["param_k_bind_n"] == float(kbn))
        & (df["param_k_bind_s"] == float(kbs))
    ]

    if subset_df.empty:
        logger.warning(f"Skipping Fano plot for kbn={kbn}, kbs={kbs}: No data.")
        return

    fig, ax = setup_plot(figsize=figsize)

    sns.histplot(
        data=subset_df,
        x="mrna_fano",
        bins=20,
        kde=True,
        color="purple",
        ax=ax,
    )
    ax.set_title(f"mRNA Fano Factor Histogram (k_bind_n={kbn}, k_bind_s={kbs})")
    ax.set_xlabel("Fano Factor (Variance / Mean)")
    ax.set_ylabel("Frequency (Runs)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/fano_hist_kbn_{kbn}_kbs_{kbs}.png"
    plt.savefig(output_file, dpi=200)
    logger.info(f"Saved Fano histogram to {output_file}")
    plt.close()


def plot_mfpt_histogram(
    df: pd.DataFrame,
    species_column: str = "starting_species",
    lifespan_column: str = "mature_lifespan_s",
    output_path: Optional[str] = None,
    title: str = "MFPT Histogram",
    figsize: tuple = (10, 6),
) -> tuple:
    """
    Plot MFPT distribution histogram with species as hue.

    Args:
        df: DataFrame with species and lifespan columns
        species_column: Column name for species/hue
        lifespan_column: Column name for lifespan data
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size

    Returns:
        Tuple of (fig, ax)
    """
    palette = get_species_palette()

    # Filter legend to only show species present in data
    present_species = df[species_column].unique()
    valid_hue_order = [s for s in palette.keys() if s in present_species]

    fig, ax = setup_plot(figsize=figsize)

    sns.histplot(
        data=df,
        x=lifespan_column,
        hue=species_column,
        palette=palette,
        hue_order=valid_hue_order,
        element="step",
        stat="density",
        common_norm=False,
        alpha=0.3,
        linewidth=2,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("MFPT (s)")
    ax.set_ylabel("Density")
    plt.tight_layout()

    if output_path:
        ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=200)
        plt.close()
        return None, None

    return fig, ax


def plot_mfpt_contour(
    df: pd.DataFrame,
    x_col: str = "param_k_bind_n",
    y_col: str = "param_k_bind_s",
    z_col: Optional[str] = None,
    output_path: Optional[str] = None,
    title: str = "MFPT Landscape (Contour)",
    figsize: tuple = (9, 7),
) -> tuple:
    """
    Create continuous contour plot of MFPT landscape.

    Args:
        df: DataFrame with x, y, z columns for tricontourf
        x_col: x parameter column name
        y_col: y parameter column name
        z_col: z value column name (auto-detected if None)
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size

    Returns:
        Tuple of (fig, ax)
    """
    fig, ax = setup_plot(figsize=figsize)

    if z_col is None:
        z_col = get_target_column(df)

    if z_col is None:
        # No data to plot
        ax.scatter(df[x_col], df[y_col], color="grey", s=100)
        ax.set_title(title + " (No binding events)")
    else:
        # Continuous surface via tricontourf
        contour_fill = ax.tricontourf(
            df[x_col],
            df[y_col],
            df[z_col],
            levels=200,
            cmap="viridis",
            extend="both",
        )

        # Highlight contours at specific levels
        highlight_lines = ax.tricontour(
            df[x_col],
            df[y_col],
            df[z_col],
            levels=[14, 16],
            colors="white",
            linewidths=0.5,
            alpha=0.6,
            linestyles="solid",
            zorder=4,
        )

        # Overlay sample coordinates
        ax.scatter(
            df[x_col],
            df[y_col],
            color="white",
            edgecolor="black",
            s=15,
            alpha=0.6,
            linewidth=0.5,
            zorder=3,
        )

        species_name = z_col.replace("MFPT_", "")
        ax.set_title(f"{title} ({species_name})")

        cbar = plt.colorbar(contour_fill, ax=ax)
        cbar.set_label("Mean First Passage Time (s)")

    ax.set_xlabel(x_col.replace("param_", ""))
    ax.set_ylabel(y_col.replace("param_", ""))
    plt.tight_layout()

    if output_path:
        ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=300)
        plt.close()
        return None, None

    return fig, ax

def plot_fano_comparison(
    df: pd.DataFrame,
    kbn1: str,
    kbs1: str,
    kbn2: str,
    kbs2: str,
    output_dir: str,
    figsize: tuple = (10, 6),
    label1: str = "Condition 1",
    label2: str = "Condition 2",
) -> None:
    """
    Overlay mRNA Fano factor histograms for two parameter coordinates.

    Args:
        df: Full statistics DataFrame
        kbn1: k_bind_n for first condition
        kbs1: k_bind_s for first condition
        kbn2: k_bind_n for second condition
        kbs2: k_bind_s for second condition
        output_dir: Directory to save plot
        figsize: Figure size
        label1: Label for first condition
        label2: Label for second condition
    """
    # Filter for both coordinates
    subset_df1 = df[
        (df["param_k_bind_n"] == float(kbn1))
        & (df["param_k_bind_s"] == float(kbs1))
    ]
    subset_df2 = df[
        (df["param_k_bind_n"] == float(kbn2))
        & (df["param_k_bind_s"] == float(kbs2))
    ]

    if subset_df1.empty and subset_df2.empty:
        logger.warning(f"No data for Fano comparison: kbn1={kbn1}, kbs1={kbs1}, kbn2={kbn2}, kbs2={kbs2}")
        return

    fig, ax = setup_plot(figsize=figsize)

    # Plot both histograms with transparency
    if not subset_df1.empty:
        ax.hist(
            subset_df1["mrna_fano"],
            alpha=0.6,
            label=label1,
            color="purple",
            density=True,
            bins=10
        )
    if not subset_df2.empty:
        ax.hist(
            subset_df2["mrna_fano"],
            alpha=0.6,
            label=label2,
            color="orange",
            density=True,
            bins=10
        )

    ax.set_title(f"mRNA Fano Factor Comparison\n{label1}: (kbn={kbn1}, kbs={kbs1}) vs {label2}: (kbn={kbn2}, kbs={kbs2})")
    ax.set_xlabel("Fano Factor (Variance / Mean)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/fano_comparison_kbn1_{kbn1}_kbs1_{kbs1}_vs_kbn2_{kbn2}_kbs2_{kbs2}.png"
    plt.savefig(output_file, dpi=200)
    logger.info(f"Saved Fano comparison plot to {output_file}")
    plt.close()


def plot_mfpt_comparison(
    kbn1: str,
    kbs1: str,
    kbn2: str,
    kbs2: str,
    output_dir: str,
    base_outdir: str,
    figsize: tuple = (10, 6),
    label1: str = "Condition 1",
    label2: str = "Condition 2",
) -> None:
    """
    Overlay MFPT distribution histograms for two parameter coordinates.

    Args:
        kbn1: k_bind_n for first condition
        kbs1: k_bind_s for first condition
        kbn2: k_bind_n for second condition
        kbs2: k_bind_s for second condition
        output_dir: Directory to save plot
        base_outdir: Base output directory containing sweep results
        figsize: Figure size
        label1: Label for first condition
        label2: Label for second condition
    """
    from .compile_data import _return_all_mfpt_trajectories
    import polars as pl

    palette = get_species_palette()
    fig, ax = setup_plot(figsize=figsize)

    # Load and process first dataset
    try:
        param_id1 = f"sweep/kbn_{kbn1}_kbs_{kbs1}"
        df_mfpt1 = _return_all_mfpt_trajectories(
            param_id=param_id1, sim_time=1000, n_sites=10, output_dir=base_outdir
        )

        if isinstance(df_mfpt1, pl.LazyFrame):
            df_mfpt1 = df_mfpt1.collect()

        df1_pandas = df_mfpt1.to_pandas()
        lifespan_col = "transient_lifespan_s"

        if lifespan_col in df1_pandas.columns:
            df1_filtered = df1_pandas.dropna(subset=[lifespan_col])
            df1_filtered = df1_filtered[df1_filtered[lifespan_col] > 0]
            df1_filtered = df1_filtered["starting_species"] == "NANOGb"

            if not df1_filtered.empty:
                ax.hist(
                    df1_filtered[lifespan_col],
                    alpha=0.5,
                    label=label1,
                    color="steelblue",
                    density=True,
                    bins=100
                )
    except FileNotFoundError:
        logger.warning(f"Data not found for condition 1: kbn={kbn1}, kbs={kbs1}")

    # Load and process second dataset
    try:
        param_id2 = f"sweep/kbn_{kbn2}_kbs_{kbs2}"
        df_mfpt2 = _return_all_mfpt_trajectories(
            param_id=param_id2, sim_time=1000, n_sites=10, output_dir=base_outdir
        )

        if isinstance(df_mfpt2, pl.LazyFrame):
            df_mfpt2 = df_mfpt2.collect()

        df2_pandas = df_mfpt2.to_pandas()
        lifespan_col = "transient_lifespan_s"

        if lifespan_col in df2_pandas.columns:
            df2_filtered = df2_pandas.dropna(subset=[lifespan_col])
            df2_filtered = df2_filtered[df2_filtered[lifespan_col] > 0]
            df2_filtered = df2_filtered["starting_species"] == "NANOGb"

            if not df2_filtered.empty:
                ax.hist(
                    df2_filtered[lifespan_col],
                    alpha=0.5,
                    label=label2,
                    color="coral",
                    density=True,
                    bins=100
                )
    except FileNotFoundError:
        logger.warning(f"Data not found for condition 2: kbn={kbn2}, kbs={kbs2}")

    ax.set_title(
        f"MFPT Distribution Comparison\n{label1}: (kbn={kbn1}, kbs={kbs1}) vs {label2}: (kbn={kbn2}, kbs={kbs2})"
    )
    ax.set_xlabel("MFPT (s)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/mfpt_comparison_kbn1_{kbn1}_kbs1_{kbs1}_vs_kbn2_{kbn2}_kbs2_{kbs2}.png"
    plt.savefig(output_file, dpi=200)
    logger.info(f"Saved MFPT comparison plot to {output_file}")
    plt.close()






