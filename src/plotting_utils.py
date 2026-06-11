"""
Shared plotting utilities for expression model.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from . import constants
from typing import Optional
from src.logger import setup_logger

logger = setup_logger(__name__)


def set_nature_style():
    """Applies Nature publication formatting guidelines to matplotlib."""
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.title_fontsize": 7,
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def setup_plot(figsize: tuple = (3.5, 3), style: str = "ticks") -> tuple:
    """
    Initialise plot with Nature theme and figure size.

    Args:
        figsize: (width, height) in inches (Nature single col: 3.5, double: 7.2)
        style: Seaborn style theme

    Returns:
        Tuple of (fig, ax)
    """
    set_nature_style()
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
    figsize: tuple = (3.5, 3),
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

    set_nature_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3))

    # Histogram
    sns.histplot(
        data=subset_df,
        x="mrna_fano",
        bins=20,
        kde=True,
        color="purple",
        ax=ax1,
    )
    ax1.set_title(f"mRNA Fano Factor Histogram")
    ax1.set_xlabel("Fano Factor (Variance / Mean)")
    ax1.set_ylabel("Frequency (Runs)")
    ax1.grid(axis="y", alpha=0.3)
    
    # Violin
    sns.violinplot(
        data=subset_df,
        y="mrna_fano",
        color="purple",
        ax=ax2,
        inner="quartile"
    )
    ax2.set_title(f"mRNA Fano Factor Violin")
    ax2.set_ylabel("Fano Factor")

    plt.suptitle(f"Fano Factor Distribution (k_bind_n={kbn}, k_bind_s={kbs})")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/fano_hist_kbn_{kbn}_kbs_{kbs}.png"
    plt.savefig(output_file, dpi=200)
    logger.info(f"Saved Fano histogram/violin to {output_file}")
    plt.close()


def plot_mfpt_histogram(
    df: pd.DataFrame,
    species_column: str = "starting_species",
    lifespan_column: str = "mature_lifespan_s",
    output_path: Optional[str] = None,
    title: str = "MFPT Contour",
    figsize: tuple = (3.5, 3),
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

    set_nature_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3))

    # Histogram
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
        ax=ax1,
    )
    ax1.set_title("Histogram")
    ax1.set_xlabel("MFPT (s)")
    ax1.set_ylabel("Density")
    
    # Violin
    sns.violinplot(
        data=df,
        x=species_column,
        y=lifespan_column,
        palette=palette,
        order=valid_hue_order,
        ax=ax2,
        inner="quartile"
    )
    ax2.set_title("Violin Plot")
    ax2.set_xlabel("Species")
    ax2.set_ylabel("MFPT (s)")

    plt.suptitle(title)
    plt.tight_layout()

    if output_path:
        ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=200)
        plt.close()
        return None, None

    return fig, (ax1, ax2)


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


def plot_binding_frequencies(
    df: pd.DataFrame,
    output_path: str = None,
    title: str = "Mean Binding Events per Site",
    figsize: tuple = (3.5, 3)
) -> None:
    """
    Plot bar chart of mean binding events per site.
    
    Args:
        df: DataFrame containing dwell_site and mean_binding_events_per_run.
        output_path: Path to save the figure.
        title: Title of the plot.
        figsize: Figure size.
    """
    set_nature_style()
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=df,
        x="dwell_site",
        y="mean_binding_events_per_run",
        color="steelblue",
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel("Site Index")
    ax.set_ylabel("Mean Binding Events (per run)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=200)
        logger.info(f"Saved binding frequencies plot to {output_path}")
        plt.close()


def plot_occupancy_timeline(
    df: pd.DataFrame,
    output_path: str = None,
    title: str = "Binding Events Timeline",
    figsize: tuple = (7.2, 3)
) -> None:
    """
    Plot timeline (raster plot) of occupancy events over time.
    
    Args:
        df: DataFrame containing site, species, start_time, end_time.
        output_path: Path to save the figure.
        title: Title of the plot.
        figsize: Figure size.
    """
    set_nature_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    palette = get_species_palette()
    
    for species, group in df.groupby("species"):
        color = palette.get(species, "black")
        ax.hlines(
            y=group["site"],
            xmin=group["start_time"],
            xmax=group["end_time"],
            color=color,
            linewidth=6,
            label=species
        )
        
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Binding Site")
    
    # Remove duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")
    
    # Ensure y-axis shows discrete sites cleanly
    if not df.empty:
        max_site = int(df["site"].max())
        ax.set_yticks(range(max_site + 1))
        
    plt.tight_layout()
    
    if output_path:
        ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved occupancy timeline plot to {output_path}")
        plt.close()

def plot_occupancy_profile(df: pd.DataFrame, output_path: str = None, title: str = "Time-Fraction Occupancy per Site", figsize: tuple = (3.5, 3)) -> None:
    """
    Plot stacked bar chart of fraction of total simulation time each site is occupied.
    
    Args:
        df: DataFrame containing dwell_site, old_species, occupancy_fraction.
        output_path: Path to save the figure.
        title: Title of the plot.
        figsize: Figure size.
    """
    set_nature_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    if df.empty:
        ax.set_title(title + " (No Data)")
    else:
        pivot_df = df.pivot(index="dwell_site", columns="old_species", values="occupancy_fraction").fillna(0)
        max_site = int(df["dwell_site"].max())
        pivot_df = pivot_df.reindex(range(max_site + 1), fill_value=0)
        
        palette = get_species_palette()
        colors = [palette.get(c, "grey") for c in pivot_df.columns]
        
        pivot_df.plot(kind="bar", stacked=True, color=colors, ax=ax, width=0.8)
        
        ax.set_title(title)
        ax.set_xlabel("Site Index")
        ax.set_ylabel("Fraction of Time Occupied")
        ax.set_ylim(0, 1)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="Species", bbox_to_anchor=(1.05, 1), loc="upper left")
        
    plt.tight_layout()
    if output_path:
        ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved occupancy profile to {output_path}")
        plt.close()

def plot_transcriptional_bursting(
    mrna_df: pd.DataFrame, 
    occupancy_df: pd.DataFrame, 
    promoter_site: int, 
    output_path: str = None, 
    title: str = "Transcriptional Bursting Timeline",
    figsize: tuple = (7.2, 2.5)
) -> None:
    """
    Plot step function of mRNA count over time, with shaded background for promoter occupancy.
    
    Args:
        mrna_df: DataFrame with 'time' and 'mRNA' columns.
        occupancy_df: Timeline DataFrame from extract_occupancy_timeline().
        promoter_site: The site index of the promoter to highlight.
        output_path: Path to save the figure.
        title: Title of the plot.
        figsize: Figure size.
    """
    fig, ax = setup_plot(figsize=figsize)
    
    if not mrna_df.empty:
        ax.step(mrna_df["time"], mrna_df["mRNA"], where="post", color="black", linewidth=2, label="mRNA Count")
    
    promoter_occupancy = occupancy_df[occupancy_df["site"] == promoter_site] if not occupancy_df.empty else pd.DataFrame()
    palette = get_species_palette()
    
    for _, row in promoter_occupancy.iterrows():
        color = palette.get(row["species"], "grey")
        ax.axvspan(row["start_time"], row["end_time"], color=color, alpha=0.3, lw=0)
        
    ax.set_title(f"{title} (Promoter at Site {promoter_site})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("mRNA Count")
    
    import matplotlib.patches as mpatches
    handles, labels = ax.get_legend_handles_labels()
    
    if not promoter_occupancy.empty:
        unique_species = promoter_occupancy["species"].unique()
        for sp in unique_species:
            handles.append(mpatches.Patch(color=palette.get(sp, "grey"), alpha=0.3))
            labels.append(f"{sp} (Promoter Bound)")
            
    if handles:
        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left")
        
    plt.tight_layout()
    if output_path:
        ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved transcriptional bursting plot to {output_path}")
        plt.close()

def plot_sliding_distance_histogram(
    df: pd.DataFrame, 
    output_path: str, 
    species_column: str = "starting_species",
    distance_column: str = "sliding_distance",
    title: str = "Sliding Distance (Antenna Effect) Histogram",
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot histogram of sliding distance distribution.
    
    Args:
        df: DataFrame containing species and sliding distance.
        output_path: Path to save the figure.
        species_column: Column name for species.
        distance_column: Column name for distance.
        title: Title of the plot.
        figsize: Figure size.
    """
    if df.empty or distance_column not in df.columns:
        fig, ax = setup_plot(figsize=figsize)
        ax.set_title(title + " (No Data)")
    else:
        set_nature_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3))
        
        palette = get_species_palette()
        present_species = df[species_column].unique()
        valid_hue_order = [s for s in palette.keys() if s in present_species]
        
        # Histogram
        sns.histplot(
            data=df,
            x=distance_column,
            hue=species_column,
            palette=palette,
            hue_order=valid_hue_order,
            multiple="dodge",
            discrete=True,
            shrink=0.8,
            ax=ax1
        )
        ax1.set_title("Histogram")
        ax1.set_xlabel("Absolute Sliding Distance (Sites)")
        ax1.set_ylabel("Count")
        
        # Violin
        sns.violinplot(
            data=df,
            x=species_column,
            y=distance_column,
            palette=palette,
            order=valid_hue_order,
            ax=ax2,
            inner="quartile"
        )
        ax2.set_title("Violin Plot")
        ax2.set_xlabel("Species")
        ax2.set_ylabel("Absolute Sliding Distance (Sites)")
        
        plt.suptitle(title)
        
    plt.tight_layout()
    if output_path:
        ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved sliding distance histogram/violin to {output_path}")
        plt.close()

def plot_survival_curve(
    df: pd.DataFrame, 
    lifespan_column: str = "mature_lifespan_s", 
    species_column: str = "starting_species", 
    output_path: str = None, 
    title: str = "Residence Time Survival Curve",
    figsize: tuple = (3.5, 3)
) -> None:
    """
    Plot log-y survival curve for residence times.
    
    Args:
        df: DataFrame containing species and lifespan.
        lifespan_column: Column name for lifespan.
        species_column: Column name for species.
        output_path: Path to save the figure.
        title: Title of the plot.
        figsize: Figure size.
    """
    import numpy as np
    fig, ax = setup_plot(figsize=figsize)
    
    if df.empty or lifespan_column not in df.columns:
        ax.set_title(title + " (No Data)")
    else:
        palette = get_species_palette()
        for species, group in df.groupby(species_column):
            color = palette.get(species, "grey")
            lifespans = np.sort(group[lifespan_column].dropna())
            if len(lifespans) == 0: continue
            
            survival_prob = 1.0 - np.arange(1, len(lifespans) + 1) / len(lifespans)
            
            # Use where="post" to keep step function appearance
            ax.step(lifespans, survival_prob, where="post", label=species, color=color, linewidth=2)
            
        ax.set_yscale("log")
        ax.set_ylabel("Survival Probability P(T > t)")
        ax.set_xlabel("Residence Time t (s)")
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        
    plt.tight_layout()
    if output_path:
        ensure_output_dir(output_path)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved survival curve to {output_path}")
        plt.close()

