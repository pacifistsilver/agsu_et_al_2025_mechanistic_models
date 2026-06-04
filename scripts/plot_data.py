import os
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from expression_model.model_compile_data import _return_all_mfpt_trajectories

SPECIES_PALETTE = {
    "SOX2b": "#1f77b4",          
    "NANOGb": "#ff7f0e",         
    "NANOGb:SOX2f": "#8CC0EB",   
    "SOX2b:NANOGb": "#d62728",   
    "NANOGb:NANOGf": "#9467bd",  
    "SOX2b:NANOGf": "#3A5B7C",   
    "NANOGb:NANOGb": "#e377c2",  
    "Heterodimer": "#DAA464"     
}



def plot_residence_time_hue(stats_df: pd.DataFrame, output_path: str):
    """Plots k_bind_n (x) vs k_bind_s (y) using mean residence time as the hue."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    
    # 1. Dynamically find a residence time column (in case NANOGb didn't form at low rates)
    duration_cols = [c for c in stats_df.columns if c.startswith('MFPT_')]
    
    if not duration_cols:
        sns.scatterplot(data=stats_df, x='param_k_bind_n', y='param_k_bind_s', color='grey', s=200)
        plt.title("No binding events occurred in these parameters")
    else:
        target_col = 'MFPT_SOX2b:NANOGf' if 'MFPT_SOX2b:NANOGf' in stats_df.columns else duration_cols[0]
        
        agg_df = stats_df.groupby(['param_k_bind_n', 'param_k_bind_s'])[target_col].mean().reset_index()

        sns.scatterplot(
            data=agg_df,
            x='param_k_bind_n',
            y='param_k_bind_s',
            hue=target_col,
            palette='viridis',
            s=200,      
            edgecolor='k'
        )
        species_name = target_col.replace('MFPT_', '')
        plt.title(f"MFPT ({species_name})")
        plt.legend(title='Mean Residence Time (s)', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xlabel("k_bind_n")
    plt.ylabel("k_bind_s")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
def plot_residence_time_heatmap(stats_df: pd.DataFrame, output_path: str):
    """Plots k_bind_n vs k_bind_s as a Heatmap using mean residence time as the Z-value."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(9, 7))
    
    # 1. Dynamically find a residence time column 
    duration_cols = [c for c in stats_df.columns if c.startswith('MFPT_')]
    
    if not duration_cols:
        # Fallback if literally nothing bound in the entire sweep
        plt.text(0.5, 0.5, 'No binding events occurred', ha='center', va='center')
        plt.title("Mean Residence Time Heatmap")
    else:
        target_col = 'MFPT_SOX2b:NANOGf' if 'MFPT_SOX2b:NANOGf' in duration_cols else duration_cols[0]
        
        # Aggregate the data
        agg_df = stats_df.groupby(['param_k_bind_n', 'param_k_bind_s'])[target_col].mean().reset_index()

        # 2. Pivot the data into a 2D Matrix for the Heatmap
        matrix_df = agg_df.pivot(index='param_k_bind_s', columns='param_k_bind_n', values=target_col)
        
        # Sort the Y-axis so the highest k_bind_s value is at the top (like a standard graph)
        matrix_df = matrix_df.sort_index(ascending=False)

        # 3. Plot the Heatmap
        sns.heatmap(
            matrix_df, 
            annot=True,         # Shows the exact number inside the square
            fmt=".2f",          # Rounds the number to 2 decimal places
            cmap="viridis",     # The color gradient
            cbar_kws={'label': 'Mean Residence Time (s)'},
            linewidths=0.5,     # Adds a clean border between squares
            linecolor='black'
        )
        
        species_name = target_col.replace('MFPT_', '')
        
        plt.title(f"Mean Residence Time ({species_name}) Parameter Landscape")
    plt.xlabel("k_bind_n")
    plt.ylabel("k_bind_s")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_individual_fanos(stats_df: pd.DataFrame, kbn_vals: list, kbs_vals: list, plot_dir: str):
    """Plots individual mRNA Fano factor histograms for each coordinate in the sweep."""
    os.makedirs(plot_dir, exist_ok=True)    
    sns.set_theme(style="whitegrid")

    for kbn in kbn_vals:
        for kbs in kbs_vals:
            # Add float() to both variables here!
            subset_df = stats_df[
                (stats_df['param_k_bind_n'] == float(kbn)) & 
                (stats_df['param_k_bind_s'] == float(kbs))
            ]
            
            plt.figure(figsize=(8, 5))
            sns.histplot(
                data=subset_df, 
                x="mrna_fano", 
                bins=20, 
                kde=True, 
                color='purple'
            )
            plt.title(f"mRNA Fano Factor Histogram (k_bind_n={kbn}, k_bind_s={kbs})")
            plt.xlabel("Fano Factor (Variance / Mean)")
            plt.ylabel("Frequency (Runs)")
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/fano_hist_kbn_{kbn}_kbs_{kbs}.png", dpi=200)
            plt.close()

def plot_individual_mfpt_histograms(kbn_vals: list, kbs_vals: list, plot_dir: str, base_outdir: str):
    """Uses _return_all_mfpt_trajectories to plot MFPT distributions for specific datapoints."""
    os.makedirs(plot_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    
    # Strict color palette so colors never shuffle between plots
    SPECIES_PALETTE = {
        "SOX2b": "#1f77b4",          
        "NANOGb": "#ff7f0e",         
        "NANOGb:SOX2f": "#8CC0EB",   
        "SOX2b:NANOGb": "#d62728",   
        "NANOGb:NANOGf": "#9467bd",  
        "SOX2b:NANOGf": "#3A5B7C",   
        "NANOGb:NANOGb": "#e377c2",  
        "Heterodimer": "#DAA464"     
    }
    
    for kbn in kbn_vals:
        for kbs in kbs_vals:
            param_id = f"sweep/kbn_{kbn}_kbs_{kbs}"
            
            df_mfpt = _return_all_mfpt_trajectories(param_id=param_id, sim_time=1000, n_sites=10, output_dir=base_outdir)
            
            if isinstance(df_mfpt, pl.LazyFrame):
                df_mfpt = df_mfpt.collect()
            
            df_pandas = df_mfpt.to_pandas()
            
            # Filter the legend to only show species that actually formed in this run
            present_species = df_pandas["starting_species"].unique()
            valid_hue_order = [s for s in SPECIES_PALETTE.keys() if s in present_species]
            
            target_col = "mature_lifespan_s" 
            df_pandas = df_pandas.dropna(subset=[target_col])
            df_pandas = df_pandas[df_pandas[target_col] > 0]


            plt.figure(figsize=(10, 6))
            sns.histplot(
                data=df_pandas, 
                x="mature_lifespan_s", 
                hue="starting_species", 
                palette=SPECIES_PALETTE,
                hue_order=valid_hue_order,
                element="step", 
                stat="density",
                common_norm=False,    
                alpha=0.3,            
                linewidth=2,

            )
            plt.title(f"MFPT Histogram (k_bind_n={kbn}, k_bind_s={kbs})")
            plt.xlabel("MFPT (s)")
            plt.ylabel("Density")
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/mfpt_hist_kbn_{kbn}_kbs_{kbs}.png", dpi=200)
            plt.close()
            
if __name__ == "__main__":
    stats_df = pl.read_parquet(snakemake.input.stats).to_pandas()
    kbn_vals_subset = snakemake.params.kbn_vals
    kbs_vals_subset = snakemake.params.kbs_vals
    base_outdir = snakemake.params.outdir
    
    plot_dir = os.path.dirname(snakemake.output.hue)

    plot_residence_time_heatmap(stats_df, snakemake.output.hue)
    plot_individual_fanos(stats_df, kbn_vals_subset, kbs_vals_subset, plot_dir)
    plot_individual_mfpt_histograms(kbn_vals_subset, kbs_vals_subset, plot_dir, base_outdir)