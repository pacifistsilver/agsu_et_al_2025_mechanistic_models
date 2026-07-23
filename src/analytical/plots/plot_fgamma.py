"""What f/gamma controls: bursts per mRNA lifetime = the NB shape parameter."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import nbinom
import twosite as ts

mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.dpi": 160, "legend.frameon": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
C = ["#2a6f97", "#c1440e", "#3d8168", "#8a5cb8", "#b08000"]
gamma = 1.0
b = 4.0                      # mean burst size, held FIXED throughout
beta = 300.0                 # fast switching -> bursty limit
k_y = b * beta


def trace(alpha, T, seed=0):
    """SSA of promoter + mRNA; alpha_s = alpha_n = alpha/2."""
    rng = np.random.default_rng(seed)
    a = alpha / 2
    ss = sn = 0
    y = 0
    t = 0.0
    ts_, ys_ = [0.0], [0]
    while t < T:
        on = ss or sn
        r = np.array([a if ss == 0 else beta, a if sn == 0 else beta,
                      k_y if on else 0.0, gamma * y])
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
        ts_.append(t)
        ys_.append(y)
    return np.array(ts_), np.array(ys_)


fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))

# ---- (a) traces at three values of f/gamma, same burst size -------------
for i, (al, lab) in enumerate([(0.3, "0.3"), (2.0, "2"), (15.0, "15")]):
    t, y = trace(al, 60.0, seed=3 + i)
    ax[0].step(t, y / (al * b), where="post", lw=0.8, color=C[i],
               label=r"$f/\gamma\approx%s$" % lab)
ax[0].set_xlim(0, 60)
ax[0].set_xlabel(r"time  ($1/\gamma$ units)")
ax[0].set_ylabel(r"$y(t)\,/\,\langle y\rangle$")
ax[0].set_title(r"(a) same burst size $b=4$, different $f/\gamma$")
ax[0].legend(fontsize=7.5, ncol=3, loc="upper center")
ax[0].set_ylim(0, 4.2)

# ---- (b) stationary distributions ---------------------------------------
for i, al in enumerate([0.2, 1.0, 3.0, 12.0]):
    a = al / 2
    f = ts.burst_frequency(a, beta, a, beta)
    P = ts.fsp_stationary(a, beta, a, beta, k_y, gamma,
                          ymax=int(80 + 40 * b * al))
    yv = np.arange(P.size)
    ax[1].plot(yv, P, color=C[i], lw=1.6, label=r"$f/\gamma=%.1f$" % (f / gamma))
    ax[1].plot(yv, nbinom.pmf(yv, f / gamma, 1 / (1 + b)), color="k",
               lw=0.8, ls="--", alpha=0.7, zorder=5)
ax[1].set_xlim(0, 90)
ax[1].set_xlabel("$y$")
ax[1].set_ylabel("$P(y)$")
ax[1].set_yscale("log")
ax[1].set_ylim(1e-4, 1)
ax[1].set_title(r"(b) exact (colour) vs NB$(f/\gamma,\,b)$ (dashed)")
ax[1].legend(fontsize=7.5)

# ---- (c) mode boundary in the (f/gamma, b) plane -------------------------
bb = np.logspace(-0.8, 2.2, 300)
ax[2].loglog(1 + 1 / bb, bb, color="k", lw=1.8)
ax[2].axvline(1, color="0.5", ls=":", lw=1.0)
ax[2].fill_betweenx(bb, 1e-2, 1 + 1 / bb, color=C[1], alpha=0.13)
ax[2].fill_betweenx(bb, 1 + 1 / bb, 1e2, color=C[0], alpha=0.13)
# exact FSP checkpoints
for bt, ft in [(0.5, 3.0054), (1.0, 2.0004), (2.0, 1.4996), (5.0, 1.1996), (20.0, 1.0499)]:
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

fig.suptitle(r"$f/\gamma$ = bursts per mRNA lifetime = shape; $b$ = size = dispersion",
             y=1.02, fontsize=11)
fig.tight_layout()
fig.savefig("./fig4_f_over_gamma.png", bbox_inches="tight")
print("done")
