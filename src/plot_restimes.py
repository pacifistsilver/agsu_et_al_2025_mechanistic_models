import numpy as np
import matplotlib.pyplot as plt
from restime_analytical import T_bound

# Set Nature-style plot settings
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

# Define a grid of alpha values from 0.01 to 1.0
alphas_s = np.logspace(-2, 0, 100)
alphas_n = np.logspace(-2, 0, 100)
A_S, A_N = np.meshgrid(alphas_s, alphas_n)

# Calculate residence times across the grid
TB = np.zeros_like(A_S)

for i in range(A_S.shape[0]):
    for j in range(A_S.shape[1]):
        a_s = A_S[i, j]
        a_n = A_N[i, j]
        TB[i, j] = T_bound(a_s, a_n)

# Create the figure
fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot T_bound
cp1 = ax1.contourf(A_S, A_N, TB, levels=50, cmap='RdBu_r')
fig.colorbar(cp1, ax=ax1, label='Effective Residence Time (s)')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$\alpha_S$ (SOX2 Binding Rate, s$^{-1}$)')
ax1.set_ylabel(r'$\alpha_N$ (NANOG Binding Rate, s$^{-1}$)')
ax1.set_title(r'Overall Bound Residence Time ($T_{bound}$) $\beta_S$ = 0.06; $\beta_N$ = 0.04')

plt.tight_layout()
plt.savefig('residence_times_landscape.png', dpi=300, bbox_inches='tight')
plt.savefig('residence_times_landscape.pdf', transparent=True, bbox_inches='tight')
print("Successfully generated and saved T_bound plots as residence_times_landscape.png and .pdf")
