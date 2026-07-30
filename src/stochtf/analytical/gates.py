"""OR / AND / additive promoter logic on the same 4-state two-site promoter."""
import numpy as np
from scipy.sparse import diags, identity, bmat, csc_matrix
from scipy.sparse.linalg import spsolve

def derived(a_s,b_s,a_n,b_n):
    ls,ln=a_s+b_s,a_n+b_n
    ps,pn=a_s/ls,a_n/ln
    return ls,ln,ps,pn,1-ps,1-pn

# promoter order 00,10,01,11
ACT = {"OR":  np.array([0,1,1,1]),
       "AND": np.array([0,0,0,1]),
       "ADD": np.array([0,1,1,2])}

def fsp(a_s,b_s,a_n,b_n,k,g,gate,ymax=800):
    act=ACT[gate]; ny=ymax+1; y=np.arange(ny)
    jumps=[(0,1,a_s),(0,2,a_n),(1,0,b_s),(1,3,a_n),(2,0,b_n),(2,3,a_s),(3,1,b_n),(3,2,b_s)]
    out=np.zeros(4)
    for u,v,r in jumps: out[u]+=r
    B=[[None]*4 for _ in range(4)]
    for s in range(4):
        birth=k*act[s]
        d=-(birth*(y<ny-1)+g*y+out[s])
        B[s][s]=diags([d,birth*np.ones(ny-1),g*y[1:]],[0,1,-1],format="csr")
    for u,v,r in jumps: B[u][v]=r*identity(ny,format="csr")
    Q=bmat(B,format="csc"); M=Q.T.tolil(); M[0,:]=1.0
    rhs=np.zeros(4*ny); rhs[0]=1.0
    return spsolve(csc_matrix(M),rhs).reshape(4,ny).sum(0)

def moments(P):
    y=np.arange(P.size); m=(y*P).sum(); v=(y*y*P).sum()-m*m
    return m,v,v/m

def analytic(a_s,b_s,a_n,b_n,k,g,gate):
    ls,ln,ps,pn,qs,qn=derived(a_s,b_s,a_n,b_n)
    Ds,Dn,D=g+ls,g+ln,g+ls+ln
    if gate=="OR":
        sig=1-qs*qn
        A=[qn**2*ps*qs, qs**2*pn*qn, ps*qs*pn*qn]
    elif gate=="AND":
        sig=ps*pn
        A=[pn**2*ps*qs, ps**2*pn*qn, ps*qs*pn*qn]
    else:
        sig=ps+pn
        A=[ps*qs, pn*qn, 0.0]
    mu=k*sig/g
    F=1+(k/sig)*(A[0]/Ds+A[1]/Dn+A[2]/D)
    return mu,F
