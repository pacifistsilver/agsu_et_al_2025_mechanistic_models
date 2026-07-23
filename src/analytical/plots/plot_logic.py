"""Fig 5: OR vs AND vs additive logic on the same two-site promoter."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import gates as G
import twosite as ts

mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.dpi": 160, "legend.frameon": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
C = {"OR": "#2a6f97", "AND": "#c1440e", "ADD": "#3d8168"}
g, mu = 1.0, 10.0          # gamma = 1, mean expression held FIXED at 10


def amps(ps, pn, qs, qn, gate):
    """Autocovariance amplitudes (A_s, A_n, A_cross) and <sigma>."""
    if gate == "OR":
        return (qn**2*ps*qs, qs**2*pn*qn, ps*qs*pn*qn), 1 - qs*qn
    if gate == "AND":
        return (pn**2*ps*qs, ps**2*pn*qn, ps*qs*pn*qn), ps*pn
    return (ps*qs, pn*qn, 0.0), ps + pn


def F_fixed_mean(a_s, b_s, a_n, b_n, gate, mu=mu, g=g):
    """F at fixed mean: k_y = gamma*mu/<sigma>."""
    ls, ln, ps, pn, qs, qn = G.derived(a_s, b_s, a_n, b_n)
    A, sb = amps(ps, pn, qs, qn, gate)
    return 1 + g*mu*(A[0]/(g+ls) + A[1]/(g+ln) + A[2]/(g+ls+ln))/sb**2


fig, ax = plt.subplots(2, 2, figsize=(9.2, 7.0))

# ---- (a) noise vs promoter switching speed, symmetric sites ------------
lam = np.logspace(-1.5, 2.5, 400)
p = 0.5
for gate in ["ADD", "OR", "AND"]:
    F = F_fixed_mean(p*lam, (1-p)*lam, p*lam, (1-p)*lam, gate)
    ax[0, 0].loglog(lam, F - 1, color=C[gate], lw=1.9, label=gate)
ax[0, 0].axhline(1, color="k", lw=0.8, ls="--")
ax[0, 0].axvline(g, color="0.5", ls=":", lw=1.0)
ax[0, 0].text(1.25, 2e-3, r"$\lambda=\gamma$", fontsize=7, color="0.4")
ax[0, 0].set_xlabel(r"promoter switching rate  $\lambda=\alpha+\beta$")
ax[0, 0].set_ylabel(r"$F-1$  (noise above Poisson)")
ax[0, 0].set_title(r"(a) fixed mean $\langle y\rangle=10$, $p_s=p_n=1/2$")
ax[0, 0].legend(fontsize=8)
ax[0, 0].set_ylim(1e-3, 1e3)

# ---- (b) where the AND-gate noise comes from --------------------------
pv = np.linspace(0.03, 0.97, 400)
lam0 = 2.0
th = (1 - pv)/pv
t1 = g*mu*th/(g + lam0)                 # site s
t2 = g*mu*th/(g + lam0)                 # site n
t3 = g*mu*th**2/(g + 2*lam0)            # coincidence cross-term
ax[0, 1].fill_between(pv, 0, t1, color="#f0c8b4", label=r"site $s$:  $\theta_s/(\gamma+\lambda_s)$")
ax[0, 1].fill_between(pv, t1, t1+t2, color="#e39a76", label=r"site $n$:  $\theta_n/(\gamma+\lambda_n)$")
ax[0, 1].fill_between(pv, t1+t2, t1+t2+t3, color="#c1440e",
                      label=r"coincidence:  $\theta_s\theta_n/(\gamma+\lambda_s+\lambda_n)$")
FOR = F_fixed_mean(pv*lam0, (1-pv)*lam0, pv*lam0, (1-pv)*lam0, "OR") - 1
ax[0, 1].plot(pv, FOR, color=C["OR"], lw=1.9, ls="--", label="OR gate (total)")
ax[0, 1].set_yscale("log")
ax[0, 1].set_ylim(1e-2, 1e3)
ax[0, 1].set_xlabel(r"site occupancy  $p_s=p_n=p$")
ax[0, 1].set_ylabel(r"$F-1$")
ax[0, 1].set_title(r"(b) AND-gate noise budget ($\lambda=2$)")
ax[0, 1].legend(fontsize=6.8, loc="upper right")

# ---- (c) AND/OR noise ratio over the occupancy plane ------------------
pp = np.linspace(0.02, 0.98, 220)
Ps, Pn = np.meshgrid(pp, pp, indexing="ij")
R = (F_fixed_mean(Ps*lam0, (1-Ps)*lam0, Pn*lam0, (1-Pn)*lam0, "AND") - 1) / \
    (F_fixed_mean(Ps*lam0, (1-Ps)*lam0, Pn*lam0, (1-Pn)*lam0, "OR") - 1)
pc = ax[1, 0].pcolormesh(pp, pp, R.T, norm=LogNorm(R.min(), R.max()),
                         cmap="RdYlBu_r", shading="auto", rasterized=True)
cs = ax[1, 0].contour(pp, pp, R.T, levels=[3, 9, 30, 100], colors="k",
                      linewidths=0.6)
ax[1, 0].clabel(cs, fmt="%g", fontsize=6.5)
ax[1, 0].plot(0.5, 0.5, "o", ms=5, mfc="w", mec="k", mew=1.3)
ax[1, 0].set_xlabel(r"$p_s$"); ax[1, 0].set_ylabel(r"$p_n$")
ax[1, 0].set_title(r"(c) $(F_{\rm AND}-1)/(F_{\rm OR}-1)$ — AND always noisier")
ax[1, 0].grid(False)
plt.colorbar(pc, ax=ax[1, 0], pad=0.02)

# ---- (d) burst-size distributions -------------------------------------
a_s, b_s, a_n, b_n, k_y = 0.7, 2.3, 0.35, 0.9, 20.0
pmf_or = ts.burst_size_pmf(a_s, b_s, a_n, b_n, k_y, mmax=200)
m = np.arange(pmf_or.size)
b_or = ts.burst_size(a_s, b_s, a_n, b_n, k_y)
b_and = k_y/(b_s + b_n)
pmf_and = (1/(1+b_and))*(b_and/(1+b_and))**m
ax[1, 1].semilogy(m, pmf_or, color=C["OR"], lw=1.7, label=r"OR (3-phase, $b=%.1f$)" % b_or)
ax[1, 1].semilogy(m, (1/(1+b_or))*(b_or/(1+b_or))**m, color=C["OR"], lw=1.0,
                  ls=":", label="geometric, same mean")
ax[1, 1].semilogy(m, pmf_and, color=C["AND"], lw=1.7,
                  label=r"AND (exactly geometric, $b=%.1f$)" % b_and)
ax[1, 1].set_xlim(0, 120); ax[1, 1].set_ylim(1e-6, 1)
ax[1, 1].set_xlabel("burst size $m$"); ax[1, 1].set_ylabel("$P(m)$")
ax[1, 1].set_title("(d) AND has a single ON state $\\Rightarrow$ geometric")
ax[1, 1].legend(fontsize=7)

fig.suptitle("Promoter logic sets the noise: coincidence (AND) is the worst case",
             y=0.995, fontsize=11)
fig.tight_layout()
fig.savefig("./fig5_promoter_logic.png", bbox_inches="tight")

print("panel (a) at lambda=2, fixed mean 10:")
for gate in ["OR", "AND", "ADD"]:
    print(f"   {gate:>4}  F = {F_fixed_mean(1.0,1.0,1.0,1.0,gate):8.4f}")
print(f"panel (d): OR CV^2 = {((m*m*pmf_or).sum()-(m*pmf_or).sum()**2)/(m*pmf_or).sum()**2:.4f}"
      f"   geometric ref {(1+b_or)/b_or:.4f}   AND (geometric) {(1+b_and)/b_and:.4f}")
print("wrote fig5")
