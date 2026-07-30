"""Exact Fano factor for ANY autonomous promoter+pool driver.

If the promoter/pool subsystem does not depend on y, then y | sigma(.) is Cox:
    Var(y) = <y> + (k^2/gamma) [ s^T diag(pi) (gamma I - Q)^-1 s  -  <s>^2/gamma ]
This holds even when the driver has BIMOLECULAR reactions (dimerisation), because
the driver is still a finite Markov chain -- we never need its moments to close.
"""
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix, identity
from scipy.sparse.linalg import spsolve

def fano_from_driver(Q, s, k, g):
    n = Q.shape[0]
    M = csc_matrix(Q).T.tolil(); M[0,:] = 1.0
    rhs = np.zeros(n); rhs[0] = 1.0
    pi = spsolve(csc_matrix(M), rhs)
    sbar = pi @ s
    mu = k*sbar/g
    R = spsolve(csc_matrix(g*identity(n) - Q), s)
    integ = (pi*s) @ R - sbar**2/g
    return mu, 1 + (k**2/(g*mu))*integ, sbar

def and_gate_driver(a_s,b_s,a_n,b_n):
    Q = lil_matrix((4,4))
    for u,v,r in [(0,1,a_s),(0,2,a_n),(1,0,b_s),(1,3,a_n),
                  (2,0,b_n),(2,3,a_s),(3,1,b_n),(3,2,b_s)]:
        Q[u,v]+=r; Q[u,u]-=r
    return Q.tocsc(), np.array([0.,0.,0.,1.])

def dimer_driver(Stot,Ntot,kp,km,kon,boff):
    """states (d,b): d free dimers, b in {0,1} promoter occupancy.
       Total S used = d + b, so S_free = Stot-d-b >= 0 (same for N).
       Only reachable states are enumerated, so Q is irreducible."""
    cap = min(Stot,Ntot)
    states = [(d,b) for b in (0,1) for d in range(cap-b+1)]
    ix = {st:i for i,st in enumerate(states)}
    n = len(states)
    Q = lil_matrix((n,n))
    def add(i,key,r):
        if r>0 and key in ix:
            j=ix[key]; Q[i,j]+=r; Q[i,i]-=r
    for (d,b) in states:
        i = ix[(d,b)]
        Sf, Nf = Stot-d-b, Ntot-d-b
        add(i,(d+1,b), kp*Sf*Nf)     # S + N -> D
        add(i,(d-1,b), km*d)         # D -> S + N
        if b==0: add(i,(d-1,1), kon*d)   # D binds promoter
        else:    add(i,(d+1,0), boff)    # D unbinds
    s = np.array([b for (d,b) in states], float)
    return Q.tocsc(), s
