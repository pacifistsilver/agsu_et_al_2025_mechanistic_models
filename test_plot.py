import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_file(filepath):
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, sep=' ', index_col=0)
    # The columns might be cell names, index is transcript ID
    # Calculate mean and variance along columns
    mean = df.mean(axis=1)
    var = df.var(axis=1)
    
    # Avoid division by zero
    mask = mean > 0
    mean = mean[mask]
    var = var[mask]
    
    cv2 = var / (mean ** 2)
    return mean, cv2

mean_129, cv2_129 = process_file('data/ochiai/GSE132589_ASEcount_G1_129.txt')
mean_cast, cv2_cast = process_file('data/ochiai/GSE132589_ASEcount_G1_CAST.txt')

print(f"129: {len(mean_129)} genes")
print(f"CAST: {len(mean_cast)} genes")

# Plotting logic
fig, ax_main = plt.subplots(figsize=(10, 8))

# Main plot: mean > 20
mask_129_high = mean_129 > 20
mask_cast_high = mean_cast > 20

ax_main.scatter(mean_129[mask_129_high], cv2_129[mask_129_high], alpha=0.5, label='G1 129', color='blue', s=10)
ax_main.scatter(mean_cast[mask_cast_high], cv2_cast[mask_cast_high], alpha=0.5, label='G1 CAST', color='orange', s=10)

ax_main.set_xlabel('Mean expression')
ax_main.set_ylabel('CV^2')
ax_main.set_title('CV^2 vs Mean (High Expression Genes: Mean > 20)')
ax_main.set_xscale('log')
ax_main.set_yscale('log')
ax_main.legend()

# Inset plot: mean <= 20
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
ax_inset = inset_axes(ax_main, width="40%", height="40%", loc='upper right', borderpad=2)

mask_129_low = mean_129 <= 20
mask_cast_low = mean_cast <= 20

ax_inset.scatter(mean_129[mask_129_low], cv2_129[mask_129_low], alpha=0.1, color='blue', s=2)
ax_inset.scatter(mean_cast[mask_cast_low], cv2_cast[mask_cast_low], alpha=0.1, color='orange', s=2)

ax_inset.set_xscale('log')
ax_inset.set_yscale('log')
ax_inset.set_title('Mean <= 20', fontsize=10)

plt.savefig('test_plot.png')
print("Plot saved to test_plot.png")
