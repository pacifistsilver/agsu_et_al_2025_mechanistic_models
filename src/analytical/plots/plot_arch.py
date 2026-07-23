"""Fig 6: where the coincidence is enforced -- promoter (AND) vs solution (heterodimer)."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import gates as G
import dimer as D

mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.dpi": 160, "legend.frameon": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
CA, CD, CX = "#c1440e", "#8a5cb8", "#3d8168"

g, mu = 1.0, 10.0            # gamma, and the FIXED mean expression
kon, bdna = 0.05, 2.0        # DNA on-rate per free molecule, DNA off-rate
kp, km = 1.0, 1.0            # dimerisation on/off (K = km/kp = 1)


def and_F(M):
    a = kon*M
    ls, ln, ps, pn, qs, qn = G.derived(a, bdna, a, bdna)
    Q, s = D.and_gate_driver(a, bdna, a, bdna)
    _, F, _ = D.fano_from_driver(Q, s, g*mu/(ps*pn), g)
    return F, ps*pn


def dim_F(M, kp=kp, km=km):
    Q, s = D.dimer_driver(M, M, kp, km, kon, bdna)
    _, _, pd = D.fano_from_driver(Q, s, 1.0, g)
    _, F, _ = D.fano_from_driver(Q, s, g*mu/pd, g)
    return F, pd


def add_F(M):
    a = kon*M
    ls, ln, ps, pn, qs, qn = G.derived(a, bdna, a, bdna)
    return 1 + g*mu*(ps*qs/(g+ls) + pn*qn/(g+ln))/(ps+pn)**2, ps+pn


fig, ax = plt.subplots(2, 2, figsize=(9.2, 7.0))

# ---- (a) noise vs monomer budget --------------------------------------
Ms = np.array([3, 4, 6, 8, 10, 14, 20, 28, 40, 56, 80, 112, 160, 220])
Fa = np.array([and_F(M)[0] for M in Ms])
Fd = np.array([dim_F(M)[0] for M in Ms])
Fx = np.array([add_F(M)[0] for M in Ms])
ax[0, 0].loglog(Ms, Fa-1, "o-", color=CA, lw=1.8, ms=4, label="AND gate (2 sites)")
ax[0, 0].loglog(Ms, Fd-1, "s-", color=CD, lw=1.8, ms=4, label="heterodimer (1 site)")
ax[0, 0].loglog(Ms, Fx-1, "^-", color=CX, lw=1.8, ms=4, label="additive monomers")
ax[0, 0].set_xlabel(r"monomer budget  $S_{tot}=N_{tot}=M$")
ax[0, 0].set_ylabel(r"$F-1$")
ax[0, 0].set_title(r"(a) matched $M$, $k_{on}$, $\beta_{DNA}$, $\langle y\rangle=10$")
ax[0, 0].legend(fontsize=7.5)

# ---- (b) the mechanism: achievable occupancy --------------------------
pa = np.array([and_F(M)[1] for M in Ms])
pd = np.array([dim_F(M)[1] for M in Ms])
px = np.array([add_F(M)[1] for M in Ms])/2      # per-site occupancy p
ax[0, 1].semilogx(Ms, pa, "o-", color=CA, lw=1.8, ms=4, label=r"AND:  $p_sp_n$")
ax[0, 1].semilogx(Ms, pd, "s-", color=CD, lw=1.8, ms=4, label=r"dimer:  $p_d$")
ax[0, 1].semilogx(Ms, px, ":", color="0.45", lw=1.4, label=r"single site $p_s$")
ax[0, 1].fill_between(Ms, pa, pd, color=CD, alpha=0.12)
ax[0, 1].set_xlabel(r"$M$"); ax[0, 1].set_ylabel(r"promoter ON fraction")
ax[0, 1].set_title(r"(b) why: $p_sp_n<\min(p_s,p_n)$ always")
ax[0, 1].legend(fontsize=7.5)
ax[0, 1].set_ylim(0, 1)

# ---- (c) crossover in dimerisation strength ---------------------------
Ks = np.logspace(-1, 4.5, 60)
M0 = 20
Fa0 = and_F(M0)[0]
Fx0 = add_F(M0)[0]
Fk = np.array([dim_F(M0, kp=1.0, km=K)[0] for K in Ks])
pk = np.array([dim_F(M0, kp=1.0, km=K)[1] for K in Ks])
ax[1, 0].loglog(Ks, (Fk-1)/(Fa0-1), color=CD, lw=2.0)
ax[1, 0].axhline(1, color="k", lw=1.0, ls="--")
ax[1, 0].axhline((Fx0-1)/(Fa0-1), color=CX, lw=1.2, ls=":")
ax[1, 0].text(0.15, (Fx0-1)/(Fa0-1)*1.15, "additive monomers", fontsize=7, color=CX)
i = np.argmin(np.abs(Fk - Fa0))
ax[1, 0].plot(Ks[i], 1.0, "o", ms=6, mfc="w", mec="k", mew=1.4, zorder=5)
ax[1, 0].fill_between(Ks, 1e-3, 1, color=CD, alpha=0.10)
ax[1, 0].fill_between(Ks, 1, 1e3, color=CA, alpha=0.10)
ax[1, 0].text(0.15, 0.14, "dimer quieter", fontsize=8, color=CD)
ax[1, 0].text(300, 30, "AND quieter", fontsize=8, color=CA)
ax[1, 0].set_xlabel(r"dimerisation dissociation constant  $K=k_-/k_+$")
ax[1, 0].set_ylabel(r"$(F_{\rm dim}-1)/(F_{\rm AND}-1)$")
ax[1, 0].set_title(r"(c) the advantage needs the complex to form ($M=20$)")
ax[1, 0].set_ylim(1e-1, 1e3)

# ---- (d) pool noise is NOT the mechanism ------------------------------
speeds = np.logspace(-3, 2, 40)
for K, col, lab in [(1.0, CD, r"$K=1$ (strong)"), (100.0, "#b08000", r"$K=100$ (weak)")]:
    r = []
    for sc in speeds:
        F, p = dim_F(M0, kp=1.0*sc, km=K*sc)
        dbar = p*bdna/kon
        lam = kon*dbar + bdna
        Fid = 1 + g*mu*((1-p)/p)/(g+lam)     # ideal telegraph, same p and lambda
        r.append((F-1)/(Fid-1))
    ax[1, 1].semilogx(speeds, r, lw=2.0, color=col, label=lab)
ax[1, 1].axhline(1, color="k", lw=1.0, ls="--")
ax[1, 1].set_ylim(0.8, 1.15)
ax[1, 1].set_xlabel(r"dimerisation speed (both $k_+,k_-$ scaled)")
ax[1, 1].set_ylabel(r"$F_{\rm dim}$ / ideal telegraph")
ax[1, 1].set_title("(d) free-dimer pool contributes almost nothing")
ax[1, 1].legend(fontsize=7.5)

fig.suptitle("Heterodimer vs AND gate: it matters where the coincidence is paid for",
             y=0.995, fontsize=11)
fig.tight_layout()
fig.savefig("./fig6_architecture.png", bbox_inches="tight")

print(f"{'M':>5}{'p AND':>9}{'p dim':>9}{'F_AND':>10}{'F_dim':>9}{'F_add':>9}{'dim/AND':>9}")
for M in [4, 10, 20, 80, 160]:
    fa, pa_ = and_F(M); fd, pd_ = dim_F(M); fx, _ = add_F(M)
    print(f"{M:5d}{pa_:9.4f}{pd_:9.4f}{fa:10.3f}{fd:9.3f}{fx:9.3f}{(fd-1)/(fa-1):9.3f}")
print(f"\ncrossover in panel (c) at K = {Ks[i]:.1f}  (p_d = {pk[i]:.4f})")
print("wrote fig6")
