import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from stochtf.plotting import PALETTE, output_path, use_paper_style

from stochtf.analytical import heterodimer as hd_model
from stochtf.analytical import monomer as m_model

BS, BN = 0.05, 0.2
AS, AN = 1.0, 0.2
gamma, k_y = 1.0, 1.0          

def M_kin(a_s, b_s, a_n, b_n, ky):
    """(tau_on, tau_off, f, b) for the monomer model."""
    return (m_model.t_on(a_s, b_s, a_n, b_n),
            m_model.t_off(a_s, b_s, a_n, b_n),
            m_model.burst_frequency(a_s, b_s, a_n, b_n),
            m_model.burst_size(a_s, b_s, a_n, b_n, ky))

def hd_kin(a_s, b_s, a_n, b_n, ky):
    """(tau_on, tau_off, f, b) for the heterodimer model."""
    return (hd_model.t_on(a_s, b_s, a_n, b_n),
            hd_model.t_off(a_s, b_s, a_n, b_n),
            hd_model.burst_frequency(a_s, b_s, a_n, b_n),
            hd_model.burst_size(a_s, b_s, a_n, b_n, ky))

use_paper_style(sans_serif="Arial")
C = PALETTE

C_M, C_HD = C[0], C[1]
fig, ax = plt.subplots(2, 3, figsize=(11, 7.2))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.52)

betas = np.logspace(-3, 0, 400)
M = np.array([M_kin(AS, b, AN, b, k_y) for b in betas]).T     # tau_on, tau_off, f, b
hd  = np.array([hd_kin(AS, b, AN, b, k_y) for b in betas]).T

# ---- (a) tau_on vs alpha
a = ax[0, 0]
a.loglog(betas, M[0], color=C_M, lw=1.8, label="M (3-state)")
a.loglog(betas, hd[0],  color=C_HD,  lw=1.8, label="HD (4-state)")
a.set_xlabel(r"$k_{off}$ (s$^{-1}$)"); a.set_ylabel(r"$\tau_{\rm on}$ (s)")
ax[0, 0].loglog(betas, 1/betas, color="#999", lw=.9, ls="--", zorder=1)
ax[0, 0].text(2e-3, 1.2e2, r"$1/k_{off}$", fontsize=8, color="#777")
ax[0, 0].legend(fontsize=8, loc="upper right")
a.legend(fontsize=7)

# ---- (b) burst frequency vs alpha
a = ax[0, 1]
a.loglog(betas, M[2]/gamma, color=C_M, lw=1.8, label="M")
a.loglog(betas, hd[2]/gamma,  color=C_HD,  lw=1.8, label="HD")

a.set_xlabel(r"$k_{off}$ (s$^{-1}$)"); a.set_ylabel("Burst Frequency")

# ---- (c) burst size vs alpha
a = ax[0, 2]
a.loglog(betas, M[3], color=C_M, lw=1.8, label="M")
a.loglog(betas, hd[3],  color=C_HD,  lw=1.8, label="HD")
a.set_xlabel(r"$k_{off}$ (s$^{-1}$)"); a.set_ylabel("Burst Size")
a.set_ylim(0)

# alphas
al = np.logspace(-3, 3, 300)
M = np.array([M_kin(a, BS, a, BN, k_y) for a in al]).T     # tau_on, tau_off, f, b
hd  = np.array([hd_kin(a, BS, a, BN, k_y) for a in al]).T

# ---- (a) tau_on vs alpha
a = ax[1, 0]
for lvl, lab in [(1/BS, r"$1/\beta_s$"), (1/BN, r"$1/\beta_n$")]:
    a.axhline(lvl, color="#999", lw=.8, ls=":")
    a.annotate(lab, xy=(0.015, lvl), xycoords=("axes fraction", "data"),
               fontsize=8, color="#777", va="bottom")

a.loglog(al, M[0], color=C_M, lw=1.8, label="M (3-state)")
a.loglog(al, hd[0],  color=C_HD,  lw=1.8, label="HD (4-state)")
a.set_xlabel(r"$k_{on}$ (s$^{-1}$)"); a.set_ylabel(r"$\tau_{\rm on}$ (s)")
a.legend(fontsize=7)

# ---- (b) burst frequency vs alpha
a = ax[1, 1]
a.loglog(al, M[2]/gamma, color=C_M, lw=1.8, label="M")
a.loglog(al, hd[2]/gamma,  color=C_HD,  lw=1.8, label="HD")
astar_hd = np.sqrt(BS * BN)
astar_m = np.sqrt(BS * BN)

ax[1, 1].plot(astar_hd, hd_model.burst_frequency(astar_hd, BS, astar_hd, BN) / gamma,
                "o", ms=5, mfc="w", mec=C[1], mew=1.4, zorder=5)
ax[1, 1].plot(astar_m, m_model.burst_frequency(astar_m, BS, astar_m, BN) / gamma,
                "o", ms=5, mfc="w", mec=C[0], mew=1.4, zorder=5)

ax[1, 1].text(0.03, 0.23, "o : $\\alpha^*=\\sqrt{\\beta_s\\beta_n}$\n",
              transform=ax[1, 1].transAxes, fontsize=7, va="top")


a.set_xlabel(r"$k_{on}$ (s$^{-1}$)"); a.set_ylabel("Burst Frequency")

# ---- (c) burst size vs alpha
a = ax[1, 2]
a.loglog(al, M[3], color=C_M, lw=1.8, label="M")
a.loglog(al, hd[3],  color=C_HD,  lw=1.8, label="HD")
a.set_xlabel(r"$k_{on}$ (s$^{-1}$)"); a.set_ylabel("Burst Size")


fig.savefig("./figures/output/fig_2_burst.svg", bbox_inches="tight")

