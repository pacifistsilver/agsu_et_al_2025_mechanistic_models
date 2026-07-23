import numpy as np
import twosite as ts

np.set_printoptions(precision=6, suppress=True)

# an asymmetric, deliberately awkward parameter set
a_s, b_s, a_n, b_n = 0.5, 0.05, 0.3, 0.2
k_y, gamma = 0.1, 0.01

print("=" * 66)
print("parameters: a_s=%.2f b_s=%.2f a_n=%.2f b_n=%.2f k_y=%.1f gamma=%.1f"
      % (a_s, b_s, a_n, b_n, k_y, gamma))
print("=" * 66)

f = ts.burst_frequency(a_s, b_s, a_n, b_n)
b = ts.burst_size(a_s, b_s, a_n, b_n, k_y)
Ton, Toff = ts.t_on(a_s, b_s, a_n, b_n), ts.t_off(a_s, b_s, a_n, b_n)

print("\n[1] mechanistic burst parameters")
print("    burst frequency f      = %.6f   (per unit time; f/gamma = %.4f)" % (f, f / gamma))
print("    mean burst size  b     = %.6f" % b)
print("    <T_on> = %.6f    <T_off> = %.6f" % (Ton, Toff))
print("    check 1/(Ton+Toff)     = %.6f" % (1 / (Ton + Toff)))

print("\n[2] consistency: <y> from moments vs from b*f/gamma")
print("    <y> (moment closure)   = %.6f" % ts.mean_y(a_s, b_s, a_n, b_n, k_y, gamma))
print("    b*f/gamma              = %.6f" % (b * f / gamma))

print("\n[3] burst-size PMF (phase-type) vs PGF vs Gillespie")
pmf = ts.burst_size_pmf(a_s, b_s, a_n, b_n, k_y, mmax=4000)
m = np.arange(pmf.size)
print("    sum(pmf)               = %.9f" % pmf.sum())
print("    mean from PMF          = %.6f" % (m * pmf).sum())
print("    mean from b = k_y*Ton  = %.6f" % b)
h = 1e-6
dB = (ts.burst_size_pgf(1.0, a_s, b_s, a_n, b_n, k_y)
      - ts.burst_size_pgf(1.0 - h, a_s, b_s, a_n, b_n, k_y)) / h
print("    mean from PGF B'(1)    = %.6f" % dB)
print("    B(1) (should be 1)     = %.9f" % ts.burst_size_pgf(1.0, a_s, b_s, a_n, b_n, k_y))
var = (m**2 * pmf).sum() - ((m * pmf).sum())**2
print("    burst size CV^2        = %.6f   (a single geometric would give %.6f)"
      % (var / (m * pmf).sum()**2, (1 + b) / b))

dur, sz, off = ts.gillespie_bursts(a_s, b_s, a_n, b_n, k_y, n_bursts=60000, seed=1)
se = sz.std(ddof=1) / np.sqrt(sz.size)
print("    Gillespie mean size    = %.6f +/- %.6f" % (sz.mean(), se))
print("    Gillespie <T_on>       = %.6f +/- %.6f" % (dur.mean(), dur.std(ddof=1) / np.sqrt(dur.size)))
print("    Gillespie <T_off>      = %.6f +/- %.6f" % (off.mean(), off.std(ddof=1) / np.sqrt(off.size)))
print("    Gillespie freq         = %.6f  (analytic %.6f)" % (1 / (dur.mean() + off.mean()), f))

print("\n[4] three geometric components (eigenvalues of the emission kernel)")
print("    decay ratios           =", ts.burst_size_geometric_rates(a_s, b_s, a_n, b_n, k_y))

print("\n[5] Fano factor: closed-form vs exact FSP stationary distribution")
P = ts.fsp_stationary(a_s, b_s, a_n, b_n, k_y, gamma, ymax=600)
yv = np.arange(P.size)
mu = (yv * P).sum()
v2 = (yv**2 * P).sum() - mu**2
print("    FSP  <y> = %.6f   Var = %.6f   F = %.6f" % (mu, v2, v2 / mu))
print("    closed-form F          = %.6f" % ts.fano(a_s, b_s, a_n, b_n, k_y, gamma))
print("    tail mass at ymax      = %.2e" % P[-1])

print("\n[6] mechanistic vs effective (what a Fano-based fit would report)")
print("    b        = %.4f     b_eff = F-1        = %.4f" % (b, ts.burst_size_eff(a_s, b_s, a_n, b_n, k_y, gamma)))
print("    f/gamma  = %.4f     f_eff/gamma        = %.4f"
      % (f / gamma, ts.burst_frequency_eff(a_s, b_s, a_n, b_n, k_y, gamma) / gamma))

print("\n[7] limit checks")
print("    single-site limit (a_n=0): f -> q_s*a_s = %.6f, analytic %.6f"
      % (b_s / (a_s + b_s) * a_s, ts.burst_frequency(a_s, b_s, 1e-12, b_n)))
print("    b (a_n=0)  = %.6f   expected k_y/b_s = %.6f"
      % (ts.burst_size(a_s, b_s, 1e-12, b_n, k_y), k_y / b_s))
print("=" * 66)
