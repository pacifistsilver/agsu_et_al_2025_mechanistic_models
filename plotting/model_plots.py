import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

file = "J:/test/nanog_w10a_stats.parquet"
out_his = "feature_histograms_w10a.png"
target_columns_wt = [
    "mean_mrna", 
    "mrna_fano", 
    "mrna_coefficient_of_variation", 
    "MFPT_NANOGb:SOX2f", 
    "MFPT_SOX2b:NANOGf", 
    "MFPT_SOX2b"
]
target_columns_w10a = [
    "mean_mrna", 
    "mrna_fano", 
    "mrna_coefficient_of_variation", 
    "MFPT_NANOGb", 
    "MFPT_SOX2b"
]
x_labels_wt = ["mrna count", "fano", "coefficient of variation", "MFPT (s)", "MFPT (s)", "MFPT (s)"]
x_labels_w10a = ["mrna count", "fano", "coefficient of variation", "MFPT (s)", "MFPT (s)"]




def plot_feature_histograms(parquet_path: str, target_columns: list, x_labels: list, save_path: str = None):
    """
    Reads a parquet file and plots histograms for specific MFPT, Dwell, and mRNA features.
    
    Args:
        parquet_path (str): Path to the input .parquet file.
        save_path (str, optional): If provided, saves the plot to this path. Otherwise, shows it.
    """
    # 1. Read the parquet file into a Pandas DataFrame
    df = pd.read_parquet(parquet_path)
    
    # 2. Define the exact columns you want to plot
    
    # Check which columns actually exist in the dataframe to avoid KeyErrors
    existing_cols = [col for col in target_columns if col in df.columns]
    missing_cols = [col for col in target_columns if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: The following columns were not found and will be skipped: {missing_cols}")
        
    num_plots = len(existing_cols)
    if num_plots == 0:
        print("No target columns found in the dataset.")
        return

    # 3. Setup the subplot grid
    # For 11 columns, a 3x4 or 4x3 grid works well. We'll dynamically calculate it.
    cols_per_row = 3
    num_rows = math.ceil(num_plots / cols_per_row)
    
    # Create the figure
    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(5 * cols_per_row, 4 * num_rows))
    axes = axes.flatten() # Flatten to make it easy to iterate over
    
    # Set Seaborn style
    sns.set_theme(style="whitegrid")
    
    # 4. Loop through the columns and plot
    for i, col_name in enumerate(existing_cols):
        ax = axes[i]
        label = x_labels[i]
        
        # Plot the histogram with a Kernel Density Estimate (KDE) line for smoothness
        sns.histplot(data=df, x=col_name, ax=ax, bins=30, kde=True, color="steelblue")
        
        # Formatting the subplot
        ax.set_title(col_name, fontsize=12, fontweight='bold')
        ax.set_xlabel(label)
        ax.set_ylabel("Counts")
        
    # 5. Clean up any empty subplots (e.g., if we have 11 plots on a 12-slot grid)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    # Adjust spacing so titles and labels don't overlap
    plt.tight_layout()
    
    # 6. Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to {save_path}")
    else:
        plt.show()

# --- Example Usage ---
plot_feature_histograms(file, target_columns_w10a, x_labels_w10a, save_path = out_his)