import os
import sys
import argparse
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# Add the project root to sys.path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.plotting_utils import get_target_column
from src.restime_analytical import T_bound

def main():
    parser = argparse.ArgumentParser(description="noise landscape")
    parser.add_argument("--input", default="output/hetmer_excl/compiled_sweep_stats.parquet", help="Path to input parquet file")
    parser.add_argument("--output_png", default="noise_landscape.png", help="Path to output PNG")
    parser.add_argument("--output_pdf", default="noise_landscape.pdf", help="Path to output PDF")
    args = parser.parse_args()

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

    # Read data
    df = pl.read_parquet(args.input).to_pandas()

    x_col = "param_k_bind_s"
    y_col = "param_k_bind_n"
    z_col = get_target_column(df, preferred="mrna_fano")
    
    
    
    if z_col is None:
        print("No target column found in data!")
        return

    fig, ax1 = plt.subplots(figsize=(8, 6))

    cp1 = ax1.tricontourf(
        df[x_col],
        df[y_col],
        df[z_col],
        cmap='RdBu_r',
        levels=10,
        extend='both'
    )

    ax1.scatter(
        df[x_col],
        df[y_col],
        color="white",
        s=15,
        edgecolor="black",
        alpha=0.1,
        linewidth=0.5,
        zorder=3,
    )
    fig.colorbar(cp1, ax=ax1, label='mRNA Noise')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$\alpha_S$ (SOX2 Binding Rate, s$^{-1}$)')
    ax1.set_ylabel(r'$\alpha_N$ (NANOG Binding Rate, s$^{-1}$)')
    ax1.set_title('mRNA Noise')

    plt.tight_layout()
    plt.savefig(args.output_png, dpi=300, bbox_inches='tight')
    plt.savefig(args.output_pdf, transparent=True, bbox_inches='tight')
    print(f"Successfully generated and saved plots as {args.output_png} and {args.output_pdf}")

if __name__ == "__main__":
    main()