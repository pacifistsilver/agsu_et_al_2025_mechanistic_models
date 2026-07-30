"""Monomer model, Figure-1 style: sweeps in (alpha_s, alpha_n) with heatmaps.

    f = q_s^S q_n^N (alpha_s S + alpha_n N),      b = gamma <y> / f
    <y> = k_y (S p_s + N p_n) / gamma
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from stochtf.plotting import PALETTE, output_path, use_paper_style

use_paper_style()
CS = PALETTE

BS, BN = 0.05, 0.2      # SOX2, NANOG dissociation rates
GAM = 1                # mRNA degradation
KY = 20*GAM               # k_y/gamma = 20
MHEAT = 1             # copy number used for the M>1 heatmaps


def f_an(a_s, a_n, S, N):
    qs, qn = BS/(a_s+BS), BN/(a_n+BN)
    return (qs**S)*(qn**N)*(a_s*S + a_n*N)


def y_an(a_s, a_n, S, N):
    ps, pn = a_s/(a_s+BS), a_n/(a_n+BN)
    return KY*(S*ps + N*pn)/GAM


def b_an(a_s, a_n, S, N):
    return GAM*y_an(a_s, a_n, S, N)/f_an(a_s, a_n, S, N)


def astar_sym(M):
    """peak of f along alpha_s = alpha_n: (2M-1)a^2 + (M-1)(bs+bn)a - bs bn = 0"""
    A, B, C = 2*M-1, (M-1)*(BS+BN), -BS*BN
    return (-B + np.sqrt(B*B - 4*A*C))/(2*A)


def sup_f(n, beta):
    """sup over alpha of a single species acting alone: (1-1/n)^(n-1) * beta"""
    return (1 - 1/n)**(n-1)*beta if n > 1 else beta


fig, ax = plt.subplots(2, 3, figsize=(13.0, 7.0))
al = np.logspace(-2.5, 0, 700)
Ms = [1, 2, 4, 8]

# ---- (a) burst frequency along the symmetric line ---------------------
for i, M in enumerate(Ms):
    ax[0, 0].loglog(al, f_an(al, al, M, M)/GAM, color=CS[i], lw=1.9,
                    label=r"$S_{tot}=N_{tot}=%d$" % M)
    a0 = astar_sym(M)
    ax[0, 0].plot(a0, f_an(a0, a0, M, M)/GAM, "o", ms=5, mfc="w", mec=CS[i], mew=1.4, zorder=5)
ax[0, 0].axhline(1, color="k", ls="--", lw=0.9)
ax[0, 0].text(1.5e-5, 1.2, r"$f/\gamma=1$", fontsize=6.5)
ax[0, 0].set_xlabel(r"$\alpha_s=\alpha_n=\alpha$")
ax[0, 0].set_ylabel(r"$f/\gamma$")
ax[0, 0].set_title(r"(a) $f=q_s^Sq_n^N(\alpha_sS+\alpha_nN)$")
ax[0, 0].legend(fontsize=7)
ax[0, 0].set_ylim(1e-4, 3e2)

# ---- (b) burst size along the symmetric line --------------------------
for i, M in enumerate(Ms):
    ax[0, 1].loglog(al, b_an(al, al, M, M), color=CS[i], lw=1.9)
    a0 = astar_sym(M)
    ax[0, 1].plot(a0, b_an(a0, a0, M, M), "o", ms=5, mfc="w", mec=CS[i], mew=1.4, zorder=5)
ax[0, 1].set_xlabel(r"$\alpha_s=\alpha_n=\alpha$")
ax[0, 1].set_ylabel(r"$b=\gamma\langle y\rangle/f$")
ax[0, 1].set_title(r"(b) burst size (circles: peak of $f$)")
ax[0, 1].set_ylim(1, 1e8)

# ---- heatmap helper ---------------------------------------------------
gg = np.logspace(-2.5, 0, 300)
As, An = np.meshgrid(gg, gg, indexing="ij")


def heat(a, Z, ttl, lab, cmap):
    pc = a.pcolormesh(gg, gg, Z.T, norm=LogNorm(Z.min(), Z.max()), cmap=cmap,
                      shading="auto", rasterized=True)
    a.contour(gg, gg, Z.T, levels=np.logspace(np.log10(Z.min()), np.log10(Z.max()), 9),
              colors="w", linewidths=0.4, alpha=0.5)
    a.set_xscale("log"); a.set_yscale("log"); a.grid(False)
    a.set_xlabel(r"$\alpha_s$ (SOX2)"); a.set_ylabel(r"$\alpha_n$ (NANOG)")
    a.set_title(ttl)
    plt.colorbar(pc, ax=a, label=lab, pad=0.02)
    return pc


# ---- (c) f heatmap, one copy each: the Figure-1 saddle -----------------
F1 = f_an(As, An, 1, 1)/GAM
heat(ax[0, 2], F1, r"(c) $f/\gamma$,  $S_{tot}=N_{tot}=1$", r"$f/\gamma$", "viridis")
ax[0, 2].axhline(BS, color="w", ls="--", lw=1.0)
ax[0, 2].axvline(BN, color="w", ls="--", lw=1.0)
ax[0, 2].plot(BN, BS, "o", ms=7, mfc="none", mec="w", mew=1.6)
ax[0, 2].text(2e-5, BS*1.4, r"$\alpha_n=\beta_s$", color="w", fontsize=7)
ax[0, 2].text(BN*1.3, 2e-5, r"$\alpha_s=\beta_n$", color="w", fontsize=7, rotation=90)
ax[0, 2].text(2e-5, 3, "saddle at $(\\beta_n,\\beta_s)$;\nmax in the corner",
              color="w", fontsize=7, va="top")

ax[1, 0].axis('off')
# ---- (e) b heatmap, M copies -----------------------------------------
BM = b_an(As, An, MHEAT, MHEAT)
heat(ax[1, 1], BM, r"(e) $b$,  $S_{tot}=N_{tot}=%d$" % MHEAT, r"$b$", "magma")
j = np.unravel_index(BM.argmin(), BM.shape)
ax[1, 1].plot(gg[j[0]], gg[j[1]], "*", ms=13, mfc="w", mec="k", mew=1.0)
ax[1, 1].text(2e-5, 3, "min $b$ coincides\nwith max $f$", color="w", fontsize=7, va="top")

# ---- (f) the ceiling on f collapses toward beta_max/e -----------------
Mv = np.arange(1, 21)
sup = np.array([max(sup_f(M, BS), sup_f(M, BN)) for M in Mv])
grid = []
for M in Mv:
    grid.append(f_an(As, An, M, M).max())
grid = np.array(grid)
ax[1, 2].semilogy(Mv, grid/GAM, "o-", color=CS[0], lw=1.9, ms=4, label="grid maximum")
ax[1, 2].semilogy(Mv, sup/GAM, "--", color=CS[1], lw=1.6,
                  label=r"$(1-1/n)^{n-1}\beta_{\max}$")
ax[1, 2].axhline(BN/np.e/GAM, color="0.4", ls=":", lw=1.2)
ax[1, 2].text(9, BN/np.e/GAM*1.25, r"$\beta_{\max}/e$", fontsize=8, color="0.35")
ax[1, 2].axhline(1, color="k", ls="--", lw=0.9)
ax[1, 2].set_xlabel(r"copies $M=S_{tot}=N_{tot}$")
ax[1, 2].set_ylabel(r"$\max_{\alpha}\ f/\gamma$")
ax[1, 2].set_title(r"(f) ceiling on $f$ saturates at $\beta_{\max}/e$")
ax[1, 2].legend(fontsize=7.5)

fig.suptitle(r"Monomer model $k_y(n_b+s_b)$ swept over $(\alpha_s,\alpha_n)$:  "
             r"$\beta_s=1/20$, $\beta_n=1/5$, $\gamma=0.01$, $k_y/\gamma=20$",
             y=0.995, fontsize=11)
fig.tight_layout()
fig.savefig(output_path("fig9_monomer_heatmaps.svg"), bbox_inches="tight")

print(f"{'M':>3}{'a* (sym)':>10}{'f*/gam sym':>12}{'grid max f/gam':>16}{'(1-1/M)^(M-1)b_max/gam':>24}")
for M in [1, 2, 3, 5, 10, 20]:
    a0 = astar_sym(M)
    print(f"{M:3d}{a0:10.5f}{f_an(a0,a0,M,M)/GAM:12.4f}"
          f"{f_an(As,An,M,M).max()/GAM:16.4f}{max(sup_f(M,BS),sup_f(M,BN))/GAM:24.4f}")
print(f"\nbeta_max/e/gamma = {BN/np.e/GAM:.4f}")
# `j` is the argmin of the b heatmap, the point starred in panel (e); the
# original referred to an undefined `i` here and raised TypeError.
print(f"min b at M={MHEAT}: (a_s,a_n)=({gg[j[0]]:.2e},{gg[j[1]]:.4f})"
      + (f"; predicted a_n = beta_n/(N-1) = {BN/(MHEAT-1):.4f}"
         if MHEAT > 1 else "; predicted a_n undefined at M=1"))
print("wrote fig9")
