"""
f/gamma = bursts per mRNA lifetime = the negative-binomial shape parameter.

Self-contained: needs only numpy, scipy, matplotlib.

Two-site OR promoter.  Sites s and n flip independently (0->1 at alpha_j,
1->0 at beta_j); mRNA is made at rate k_y whenever at least one site is bound
and degrades at rate gamma per molecule.

Throughout we hold the mean burst size b fixed and vary only f/gamma, to show
that the two are orthogonal knobs: b sets the dispersion (F = 1 + b), while
f/gamma sets the shape of P(y).

Run:  python plot_fgamma_standalone.py
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity, bmat, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import nbinom

OUT = "fig4_f_over_gamma.svg"

mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.dpi": 160, "legend.frameon": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, 
                     "font.sans-serif": "Arial"})
C = ["#2a6f97", "#c1440e", "#3d8168", "#8a5cb8", "#b08000"]

# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------
# promoter state order: 00, 10, 01, 11 ;  sigma = OR = [0,1,1,1]
ACT = np.array([0, 1, 1, 1])


def burst_frequency(a_s, b_s, a_n, b_n):
    """f = q_s q_n (alpha_s + alpha_n),  q_j = beta_j/(alpha_j+beta_j)."""
    q_s = b_s / (a_s + b_s)
    q_n = b_n / (a_n + b_n)
    return q_s * q_n * (a_s + a_n)


def burst_size(a_s, b_s, a_n, b_n, k_y):
    """b = k_y <T_on> = k_y (1 - q_s q_n) / [q_s q_n (alpha_s + alpha_n)]."""
    q_s = b_s / (a_s + b_s)
    q_n = b_n / (a_n + b_n)
    r = q_s * q_n
    return k_y * (1 - r) / (r * (a_s + a_n))


def fsp_stationary(a_s, b_s, a_n, b_n, k_y, gamma, ymax=400):
    """Exact stationary P(y) by finite state projection (vectorised)."""
    ny = ymax + 1
    y = np.arange(ny)
    # promoter transition list: (from, to, rate)
    jumps = [(0, 1, a_s), (0, 2, a_n), (1, 0, b_s), (1, 3, a_n),
             (2, 0, b_n), (2, 3, a_s), (3, 1, b_n), (3, 2, b_s)]
    out = np.zeros(4)
    for u, v, r in jumps:
        out[u] += r

    blocks = [[None] * 4 for _ in range(4)]
    for s in range(4):
        birth = k_y * ACT[s]
        d = -(birth * (y < ny - 1) + gamma * y + out[s])
        blocks[s][s] = diags([d, birth * np.ones(ny - 1), gamma * y[1:]],
                             offsets=[0, 1, -1], format="csr")
    for u, v, r in jumps:
        blocks[u][v] = r * identity(ny, format="csr")

    Q = bmat(blocks, format="csc")
    M = Q.T.tolil()
    M[0, :] = 1.0                       # replace a row by the normalisation
    rhs = np.zeros(4 * ny)
    rhs[0] = 1.0
    return spsolve(csc_matrix(M), rhs).reshape(4, ny).sum(axis=0)


def trace(alpha, beta, k_y, gamma, T, seed=0):
    """Gillespie SSA of promoter + mRNA, with alpha_s = alpha_n = alpha/2."""
    rng = np.random.default_rng(seed)
    a = alpha / 2
    ss = sn = 0
    y = 0
    t = 0.0
    tt, yy = [0.0], [0]
    while t < T:
        on = ss or sn
        r = np.array([a if ss == 0 else beta,
                      a if sn == 0 else beta,
                      k_y if on else 0.0,
                      gamma * y])
        tot = r.sum()
        t += rng.exponential(1 / tot)
        j = rng.choice(4, p=r / tot)
        if j == 0:
            ss ^= 1
        elif j == 1:
            sn ^= 1
        elif j == 2:
            y += 1
        else:
            y -= 1
        tt.append(t)
        yy.append(y)
    return np.array(tt), np.array(yy)


def mode_threshold(b, gamma=1.0, beta=400.0, iters=24):
    """Bisect for the f/gamma at which the mode of P(y) leaves y = 0.
    Analytic prediction (bursty limit): f/gamma = 1 + 1/b."""
    k_y = b * beta
    lo, hi = 0.2, 8.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        a = mid / 2
        P = fsp_stationary(a, beta, a, beta, k_y, gamma,
                           ymax=int(80 + 25 * (b + 1) * mid))
        if P[1] > P[0]:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------
gamma = 1.0
b = 4.0            # mean burst size, held FIXED in panels (a) and (b)
beta = 300.0       # beta >> gamma  ->  bursty limit, so P(y) should be NB
k_y = b * beta

fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))

# ---- (a) traces at three values of f/gamma, identical burst size ------
for i, (al, lab) in enumerate([(0.3, "0.3"), (2.0, "2"), (15.0, "15")]):
    t, y = trace(al, beta, k_y, gamma, 60.0, seed=3 + i)
    ax[0].step(t, y / (al * b), where="post", lw=0.8, color=C[i],
               label=r"$f/\gamma\approx%s$" % lab)
ax[0].set_xlim(0, 60)
ax[0].set_ylim(0, 4.2)
ax[0].set_xlabel(r"time  ($1/\gamma$ units)")
ax[0].set_ylabel(r"$y(t)\,/\,\langle y\rangle$")
ax[0].set_title(r"(a) same burst size $b=4$, different $f/\gamma$")
ax[0].legend(fontsize=7.5, ncol=3, loc="upper center")

# ---- (b) stationary distributions vs the NB prediction ----------------
for i, al in enumerate([0.2, 1.0, 3.0, 12.0]):
    a = al / 2
    f = burst_frequency(a, beta, a, beta)
    P = fsp_stationary(a, beta, a, beta, k_y, gamma, ymax=int(80 + 40 * b * al))
    yv = np.arange(P.size)
    ax[1].plot(yv, P, color=C[i], lw=1.6, label=r"$f/\gamma=%.1f$" % (f / gamma))
    ax[1].plot(yv, nbinom.pmf(yv, f / gamma, 1 / (1 + b)),
               color="k", lw=0.8, ls="--", alpha=0.7, zorder=5)
ax[1].set_xlim(0, 90)
ax[1].set_yscale("log")
ax[1].set_ylim(1e-4, 1)
ax[1].set_xlabel("$y$")
ax[1].set_ylabel("$P(y)$")
ax[1].set_title(r"(b) exact (colour) vs NB$(f/\gamma,\,b)$ (dashed)")
ax[1].legend(fontsize=7.5)

# ---- (c) where the mode leaves zero ----------------------------------
bb = np.logspace(-0.8, 2.2, 300)
ax[2].loglog(1 + 1 / bb, bb, color="k", lw=1.8)
ax[2].axvline(1, color="0.5", ls=":", lw=1.0)
ax[2].fill_betweenx(bb, 1e-2, 1 + 1 / bb, color=C[1], alpha=0.13)
ax[2].fill_betweenx(bb, 1 + 1 / bb, 1e2, color=C[0], alpha=0.13)

print(f"{'b':>7}{'1 + 1/b':>12}{'exact (FSP)':>14}")
for bt in [0.5, 1.0, 2.0, 5.0, 20.0]:
    ft = mode_threshold(bt, gamma)
    print(f"{bt:7.1f}{1 + 1/bt:12.4f}{ft:14.4f}")
    ax[2].plot(ft, bt, "o", ms=5, mfc="w", mec="k", mew=1.3, zorder=6)

ax[2].text(0.12, 40, "mode at $y=0$\n(monotone decay,\nsilent fraction large)",
           fontsize=7.5, color=C[1], ha="left")
ax[2].text(3.2, 0.4, "interior mode\n(unimodal peak)", fontsize=7.5, color=C[0])
ax[2].set_xlim(1e-1, 1e2)
ax[2].set_ylim(bb[0], bb[-1])
ax[2].set_xlabel(r"$f/\gamma$")
ax[2].set_ylabel("$b$")
ax[2].set_title(r"(c) shape boundary: $f/\gamma = 1 + 1/b$")
ax[2].text(0.04, 0.04, "o : exact FSP", transform=ax[2].transAxes, fontsize=7)

fig.suptitle(r"$f/\gamma$ = bursts per mRNA lifetime = shape;  "
             r"$b$ = burst size = dispersion", y=1.02, fontsize=11)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
