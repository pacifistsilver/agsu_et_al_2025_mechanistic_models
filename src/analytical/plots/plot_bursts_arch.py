"""Fig 7: burst frequency and burst size across promoter architectures."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import bursts as BB
import dimer as D

mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.dpi": 160, "legend.frameon": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
C = {"OR": "#2a6f97", "AND": "#c1440e", "ADD": "#3d8168", "DIM": "#8a5cb8"}
LB = {"OR": "OR (2 sites)", "AND": "AND (2 sites)",
      "ADD": "additive (2 sites)", "DIM": "heterodimer (1 site)"}

g, lam, k_y = 1.0, 2.0, 20.0     # gamma, promoter switching rate, transcription rate
MU = 10.0                        # fixed mean for the fixed-mean panels

# closed forms, symmetric sites: p_s = p_n = p, alpha = p*lam, beta = q*lam
def f_of(p, arch, lam=lam):
    q = 1 - p
    return {"OR": 2*p*q*q*lam, "ADD": 2*p*q*q*lam,
            "AND": 2*p*p*q*lam, "DIM": p*q*lam}[arch]

def sig_of(p, arch):
    q = 1 - p
    return {"OR": 1-q*q, "ADD": 2*p, "AND": p*p, "DIM": p}[arch]

def b_of(p, arch, k=k_y):        # b = gamma<y>/f  with <y> = k<sigma>/gamma
    return k*sig_of(p, arch)/f_of(p, arch)


fig, ax = plt.subplots(2, 3, figsize=(13.2, 7.0))
pv = np.linspace(0.02, 0.98, 500)

# ---- (a) burst frequency ---------------------------------------------
for arch in ["OR", "AND", "ADD", "DIM"]:
    ls = "--" if arch == "ADD" else "-"
    lw = 2.6 if arch == "ADD" else 1.8
    ax[0, 0].plot(pv, f_of(pv, arch)/g, ls, color=C[arch], lw=lw, label=LB[arch])
for p0, arch in [(1/3, "OR"), (2/3, "AND"), (0.5, "DIM")]:
    ax[0, 0].plot(p0, f_of(p0, arch)/g, "o", ms=5, mfc="w", mec=C[arch], mew=1.4, zorder=5)
ax[0, 0].axvline(0.5, color="0.5", ls=":", lw=1.0)
ax[0, 0].set_xlabel("site occupancy $p$")
ax[0, 0].set_ylabel(r"burst frequency  $f/\gamma$")
ax[0, 0].set_title(r"(a) $f$: OR and additive coincide")
ax[0, 0].legend(fontsize=7)

# ---- (b) burst size at fixed k_y -------------------------------------
for arch in ["OR", "AND", "ADD", "DIM"]:
    ax[0, 1].semilogy(pv, b_of(pv, arch), color=C[arch], lw=1.8, label=LB[arch])
ax[0, 1].axvline(0.5, color="0.5", ls=":", lw=1.0)
for arch in ["OR", "AND", "ADD", "DIM"]:
    ax[0, 1].plot(0.5, b_of(0.5, arch), "o", ms=4.5, mfc="w", mec=C[arch], mew=1.4, zorder=5)
ax[0, 1].set_xlabel("site occupancy $p$")
ax[0, 1].set_ylabel(r"mean burst size  $b$")
ax[0, 1].set_title(r"(b) $b$ at fixed $k_y=20$: they differ 4-fold at $p=1/2$")
ax[0, 1].set_ylim(1, 1e4)

# ---- (c) trade-off plane ---------------------------------------------
for arch in ["OR", "AND", "ADD", "DIM"]:
    ax[0, 2].loglog(f_of(pv, arch)/g, b_of(pv, arch), color=C[arch], lw=1.8)
    ax[0, 2].plot(f_of(0.5, arch)/g, b_of(0.5, arch), "o", ms=5,
                  mfc="w", mec=C[arch], mew=1.4, zorder=5)
bb = np.logspace(0, 4.5, 40)
for yv in [1, 10, 100]:
    ax[0, 2].loglog(yv/bb, bb, color="0.6", ls="--", lw=0.7, zorder=0)
    ax[0, 2].text(yv/2.2e3, 2.2e3, r"$\langle y\rangle=%g$" % yv, fontsize=6.5,
                  color="0.45", rotation=-45, ha="left", va="top")
ax[0, 2].set_xlim(1e-3, 3)
ax[0, 2].set_ylim(1, 3e3)
ax[0, 2].set_xlabel(r"$f/\gamma$"); ax[0, 2].set_ylabel("$b$")
ax[0, 2].set_title(r"(c) trade-off; circles at $p=1/2$ (equal $f$)")

# ---- (d) burst size distributions at p = 1/2 -------------------------
a = b = 0.5*lam
Q4 = lil_matrix((4, 4))
for u, v, r in [(0,1,a),(0,2,a),(1,0,b),(1,3,a),(2,0,b),(2,3,a),(3,1,b),(3,2,b)]:
    Q4[u, v] += r; Q4[u, u] -= r
Q4 = Q4.tocsc()
Q2 = lil_matrix((2, 2))
Q2[0,1] = a; Q2[0,0] = -a; Q2[1,0] = b; Q2[1,1] = -b
Q2 = Q2.tocsc()
kvs = {"OR": (Q4, [0, k_y, k_y, k_y]), "AND": (Q4, [0, 0, 0, k_y]),
       "ADD": (Q4, [0, k_y, k_y, 2*k_y]), "DIM": (Q2, [0, k_y])}
print(f"{'arch':>5}{'f/gamma':>10}{'b':>10}{'CV^2':>9}{'geom CV^2':>11}{'geometric?':>12}")
for arch in ["AND", "DIM", "OR", "ADD"]:
    Qa, kv = kvs[arch]
    f, bmean, pmf, my = BB.burst_stats(Qa, kv, g, mmax=3000)
    m = np.arange(pmf.size)
    cv2 = ((m*m*pmf).sum() - bmean**2)/bmean**2
    geo = (1/(1+bmean))*(bmean/(1+bmean))**m
    ax[1, 0].semilogy(m, pmf, color=C[arch], lw=1.8, label=LB[arch] + r"  ($b=%.0f$)" % bmean)
    print(f"{arch:>5}{f/g:10.4f}{bmean:10.3f}{cv2:9.4f}{(1+bmean)/bmean:11.4f}"
          f"{np.abs(pmf-geo).max():12.2e}")
ax[1, 0].set_xlim(0, 200); ax[1, 0].set_ylim(1e-5, 0.3)
ax[1, 0].set_xlabel("burst size $m$"); ax[1, 0].set_ylabel("$P(m)$")
ax[1, 0].set_title(r"(d) burst-size laws at $p=1/2$")
ax[1, 0].legend(fontsize=7)

# ---- (e) burst size at FIXED MEAN -> tracks the noise -----------------
for arch in ["OR", "AND", "ADD", "DIM"]:
    ax[1, 1].semilogy(pv, g*MU/f_of(pv, arch), color=C[arch], lw=1.8, label=LB[arch])
ax[1, 1].axvline(0.5, color="0.5", ls=":", lw=1.0)
ax[1, 1].set_xlabel("site occupancy $p$")
ax[1, 1].set_ylabel(r"$b=\gamma\langle y\rangle/f$")
ax[1, 1].set_title(r"(e) fixed $\langle y\rangle=10$: $b$ is the noise ($F\!-\!1\approx b$)")
ax[1, 1].set_ylim(3, 1e4)

# ---- (f) vs monomer budget: AND vs true heterodimer -------------------
kon, bdna = 0.05, 2.0
Ms = np.array([4, 6, 8, 10, 14, 20, 28, 40, 56, 80, 112, 160])
fa, ba, fd, bd = [], [], [], []
for M in Ms:
    al = kon*M
    Qa = lil_matrix((4, 4))
    for u, v, r in [(0,1,al),(0,2,al),(1,0,bdna),(1,3,al),
                    (2,0,bdna),(2,3,al),(3,1,bdna),(3,2,bdna)]:
        Qa[u, v] += r; Qa[u, u] -= r
    ps = al/(al+bdna)
    f1, b1, _, _ = BB.burst_stats(Qa.tocsc(), [0, 0, 0, g*MU/(ps*ps)], g, mmax=2000)
    Qd, s = D.dimer_driver(M, M, 1.0, 1.0, kon, bdna)
    _, _, pd = D.fano_from_driver(Qd, s, 1.0, g)
    f2, b2, _, _ = BB.burst_stats(Qd, (g*MU/pd)*s, g, mmax=2000)
    fa.append(f1); ba.append(b1); fd.append(f2); bd.append(b2)
ax[1, 2].loglog(Ms, fa, "o-", color=C["AND"], lw=1.8, ms=4, label=r"AND  $f/\gamma$")
ax[1, 2].loglog(Ms, fd, "s-", color=C["DIM"], lw=1.8, ms=4, label=r"dimer  $f/\gamma$")
ax[1, 2].loglog(Ms, ba, "o--", color=C["AND"], lw=1.4, ms=3.5, mfc="w", label=r"AND  $b$")
ax[1, 2].loglog(Ms, bd, "s--", color=C["DIM"], lw=1.4, ms=3.5, mfc="w", label=r"dimer  $b$")
ax[1, 2].set_xlabel(r"monomer budget $M$")
ax[1, 2].set_title(r"(f) fixed $\langle y\rangle=10$: dimer bursts more often, smaller")
ax[1, 2].legend(fontsize=7, ncol=2)

fig.suptitle("Burst frequency and burst size across promoter architectures "
             r"($\lambda=2$, $\gamma=1$)", y=0.995, fontsize=11)
fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/fig7_burst_architectures.png", bbox_inches="tight")

print("\nat p=1/2 (closed forms):")
for arch in ["OR", "AND", "ADD", "DIM"]:
    print(f"   {arch:>4}  f/gamma={f_of(0.5,arch)/g:.4f}   b={b_of(0.5,arch):7.2f}")
print("\nM=20, fixed mean:  AND f=%.4f b=%.2f | dimer f=%.4f b=%.2f"
      % (fa[5], ba[5], fd[5], bd[5]))
print("wrote fig7")
