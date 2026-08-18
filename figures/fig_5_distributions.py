import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, TwoSlopeNorm, LinearSegmentedColormap, Normalize

from stochtf.plotting import use_paper_style
from stochtf.cme import stationary as st
from fig_4_noise import both_models

use_paper_style(sans_serif="Arial")
mpl.rcParams.update({
    "font.family": "DejaVu Sans", 
    "font.size": 8,
    "axes.linewidth": .7, 
    "axes.edgecolor": "#3a3a3a",
    "legend.frameon": False, 
    "figure.dpi": 160
})

BS0, BN0 = 0.05, 0.20        # reference unbinding rates
GAM, KY = 1.0, 20.0

C_M, C_HD = "#c0562b", "#1f5f8b"
LBL_M = "M"                  # exclusive-binding monomer
RATE = r"(s$^{-1}$)"
NGRID = 50
build = lambda x, y: (x, BS0, y, BN0)

def stats(P):
    y = np.arange(P.size)
    mean = P @ y
    var = P @ (y - mean)**2
    return mean, var/mean, np.sqrt(var)/mean

def heat(a, Z, title, cmap, norm, cb_label, levels=None):
    # Using global X, Y as in original script
    im = a.pcolormesh(X, Y, Z, cmap=cmap, norm=norm,
                      shading="auto", rasterized=True)
    if levels is not None:
        cs = a.contour(X, Y, Z, levels=levels, colors="w",
                       linewidths=.4, alpha=.7)
        a.clabel(cs, fmt="%g", fontsize=5.5, inline=True)
    a.set_xscale("log")
    a.set_yscale("log")
    cb = fig.colorbar(im, ax=a, pad=.02, fraction=.046)
    cb.set_label(cb_label, fontsize=7)
    cb.ax.tick_params(labelsize=6)
    a.set_title(title, fontsize=9)

# 1. Define the grid and calculate the Total Variation heatmap ONCE
al = np.logspace(-2.5, 2.5, NGRID)
X, Y = np.meshgrid(al, al)
TV = np.zeros_like(X)

print("Calculating Total Variation Heatmap...")
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Ph, Pm, _, _ = both_models(build(X[i, j], Y[i, j]))
        TV[i, j] = 0.5 * np.abs(Ph - Pm).sum()

# 2. Define four points corresponding to decreasing total variation (moving bottom-left to top-right)
dist_pts = [
    (0.01, 0.01),   # High TV (bursty regime)
    (0.1, 0.4),     # Moderate-High TV
    (1.0, 1.0),     # Low-Moderate TV
    (100.0, 100.0)    # Low TV (averaged regime)
]

dists = []
for x, y in dist_pts:
    Ph, Pm, _, _ = both_models(build(x, y), floor=120)
    tv = 0.5 * np.abs(Ph - Pm).sum()
    dists.append((x, y, Ph, Pm, tv))

# 3. Setup Layout: 1 large plot (heatmap) left, 2x2 grid (distributions) right
fig = plt.figure(figsize=(12, 5))
gs = GridSpec(2, 4, figure=fig, wspace=0.5, hspace=0.6)

ax_heat = fig.add_subplot(gs[0:2, 0:2])
ax_dists = [
    fig.add_subplot(gs[0, 2]),
    fig.add_subplot(gs[0, 3]),
    fig.add_subplot(gs[1, 2]),
    fig.add_subplot(gs[1, 3])
]

labels = ['A', 'B', 'C', 'D']
markers = ['o', 's', '^', 'D']
vmax = np.max(TV)

orig_cmap = plt.get_cmap("RdBu_r")
white_to_red = orig_cmap(np.linspace(0.5, 1.0, 256))
half_RdBu = LinearSegmentedColormap.from_list("half_RdBu", white_to_red)

# 4. Plot Heatmap
heat(ax_heat, TV, 
     title=rf"Total Variation $d_{{\rm TV}}(P_{{\rm HD}}, P_{{\rm {LBL_M}}})$",
     cmap=half_RdBu, norm=Normalize(vmin=0, vmax=np.max(TV)), cb_label=r"$d_{\rm TV}$", levels=[0.025, 0.05, 0.1, 0.2])

ax_heat.set_xlabel(r"$k_{on, s}$ " + RATE)
ax_heat.set_ylabel(r"$k_{on, n}$ " + RATE)


# 5. Plot Distributions & Mark Heatmap
for k, (x, y, Ph, Pm, tv) in enumerate(dists):
    # Plot point on the heatmap
    ax_heat.scatter(x, y, color='white', edgecolor='black', marker=markers[k], s=40, zorder=5)
    ax_heat.text(x * 1.3, y * 1.3, labels[k], color='white', fontsize=9, fontweight='bold', zorder=5)
    
    # Plot distributions
    a = ax_dists[k]
    yy = np.arange(Ph.size)
    a.plot(yy, Pm, color=C_M, lw=1.6, label=LBL_M)
    a.fill_between(yy, Pm, color=C_M, alpha=.18)
    a.plot(yy, Ph, color=C_HD, lw=1.6, label="HD")
    a.fill_between(yy, Ph, color=C_HD, alpha=.18)
    
    a.set_yscale("log")
    a.set_ylim(1e-5, 1.0)
    a.set_xlim(0, min(Ph.size-1, 60))
    a.set_xlabel("mRNA count")
    a.set_ylabel("$P(mRNA)$")
    
    # Formatting specific distribution plots
    a.set_title(f"Pt {labels[k]}: $k_{{on}}=({x}, {y})$\n$d_{{TV}}={tv:.3f}$", fontsize=8)
    if k == 0:
        a.legend(fontsize=7, loc="lower left")

os.makedirs("./figures/output", exist_ok=True)
out = f"./figures/output/fig_5_distributions"
fig.savefig(out + ".svg", bbox_inches="tight", facecolor="w")
plt.close(fig)