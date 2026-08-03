"""
Check the elementary story:
    Var(y) = <y> + Var(mu),   mu = integral of past transcription rate, weighted by survival
against direct Gillespie simulation of the full system.
"""
import numpy as np

def simulate(a_s, b_s, a_n, b_n, k_y, gamma, logic="or", T=4e6, seed=0):
    rng = np.random.default_rng(seed)
    ss = sn = 0; y = 0; t = 0.0
    ts, ys = [], []
    while t < T:
        kk = k_y*(ss or sn) if logic == "or" else k_y*ss + k_y*sn
        rates = np.array([a_s if ss==0 else b_s, a_n if sn==0 else b_n,
                          kk, gamma*y])
        tot = rates.sum()
        dt = rng.exponential(1/tot)
        ts.append(dt); ys.append(y)          # y held constant over dt
        t += dt
        j = rng.choice(4, p=rates/tot)
        if j==0: ss ^= 1
        elif j==1: sn ^= 1
        elif j==2: y += 1
        else: y -= 1
    ts = np.array(ts); ys = np.array(ys, float)
    w = ts/ts.sum()                          # time-weighted (not event-weighted!)
    m1 = (w*ys).sum()
    var = (w*(ys-m1)**2).sum()
    return m1, var, var/m1

def theory(a_s, b_s, a_n, b_n, k_y, gamma, logic="or"):
    ls, ln = a_s+b_s, a_n+b_n
    ps, qs = a_s/ls, b_s/ls
    pn, qn = a_n/ln, b_n/ln
    if logic == "or":
        kbar = k_y*(1-qs*qn)
        # C(tau) = sum A_j exp(-lam_j tau)
        terms = [(k_y**2*qn**2*ps*qs, ls), (k_y**2*qs**2*pn*qn, ln),
                 (k_y**2*ps*qs*pn*qn, ls+ln)]
    else:
        kbar = k_y*(ps+pn)
        terms = [(k_y**2*ps*qs, ls), (k_y**2*pn*qn, ln)]
    mean = kbar/gamma
    # Var(mu) = sum_j A_j / (gamma (gamma + lam_j))     <- the double integral
    var_mu = sum(A/(gamma*(gamma+lam)) for A, lam in terms)
    return mean, mean + var_mu, 1 + var_mu/mean

print("Elementary decomposition  Var(y) = <y> + Var(mu)   vs   Gillespie\n")
cases = [("HD", "or",  0.05, 0.05, 0.2, 0.2, 0.5, 0.05),
         ("M ", "add", 0.05, 0.05, 0.2, 0.2, 0.5, 0.05),
         ("HD", "or",  0.7, 0.3, 0.02, 0.9, 2.0, 0.4)]
for name, logic, a_s, b_s, a_n, b_n, k_y, g in cases:
    tm, tv, tf = theory(a_s, b_s, a_n, b_n, k_y, g, logic)
    sm, sv, sf = simulate(a_s, b_s, a_n, b_n, k_y, g, logic, T=3e6, seed=3)
    print(f"{name} ({logic:3s})  theory: <y>={tm:8.4f}  Var={tv:9.4f}  F={tf:.4f}")
    print(f"{'':10}sim   : <y>={sm:8.4f}  Var={sv:9.4f}  F={sf:.4f}\n")

# the single integral that does all the work
print("Check  int_0^inf int_0^inf e^{-g(u+v)} e^{-lam|u-v|} du dv = 1/(g(g+lam))")
from scipy.integrate import dblquad
for g, lam in [(0.05, 0.25), (0.4, 1.0), (0.001, 0.05)]:
    num = dblquad(lambda v, u: np.exp(-g*(u+v))*np.exp(-lam*abs(u-v)),
                  0, 200/g, 0, 200/g)[0]
    print(f"  gamma={g:6.3f} lambda={lam:5.2f}   numeric={num:.8g}   formula={1/(g*(g+lam)):.8g}")