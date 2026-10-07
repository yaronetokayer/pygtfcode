"""
Helpers for Kp computations.

Supplied van den Bosch 2026 note, Eq.12

x is 1D dispersion / w, not pairwise speed / w. The finite epsilon is
retained literally; no silent renormalization of individual Kp values.
"""
import math
import numpy as np
from numba import njit

COEFF=np.array([[8.,.0339848,.37,.63],[24.,.251115,.41,.71],
                [48.,.682602,.42,.74],[80.,1.32953,.43,.76]])

@njit(cache=True)
def softplus(z):
    return max(z,0.)+math.log1p(math.exp(-abs(z)))

@njit(cache=True)
def kp_log_slope(x,p):
    """Return log(Kp), dlog(Kp)/dlog(x); supports p=3,5,7,9."""
    if p not in (3,5,7,9) or x<0 or not math.isfinite(x):
        raise ValueError('Require finite x>=0 and p in (3,5,7,9)')
    p0,p1,p2,p3=COEFF[(p-3)//2]
    logx4=4*math.log(x) if x>0 else -math.inf
    logeps=math.log(1e-12)
    logs=max(logx4,logeps)+math.log1p(math.exp(-abs(logx4-logeps)))
    z=p2*(logs+math.log(p1)); log1pz=softplus(z)
    logA=logs+math.log(p0*p2/1.5)-math.log(log1pz)
    y=p3*logA
    logK=-softplus(y)/p3
    logistic=math.exp(-softplus(-y))
    dlogA_dlogs=1-p2*math.exp(-softplus(-z))/log1pz
    dlogs_dlogx=4*math.exp(logx4-logs)
    return logK,-logistic*dlogA_dlogs*dlogs_dlogx

@njit(cache=True)
def factors(T,what,order=1):
    """K_L, K_S and their dlog/dlog(T). order2 is constant-limit normalized."""
    if math.isinf(what):return 1.,1.,0.,0.
    x=math.sqrt(T)/what
    l5,s5=kp_log_slope(x,5);k5=math.exp(l5);s5*=.5
    if order==1:return k5,k5,s5,s5
    if order!=2:raise ValueError('order must be 1 or 2')
    l7,s7=kp_log_slope(x,7);l9,s9=kp_log_slope(x,9);s7*=.5;s9*=.5
    # Factor by K5 to avoid underflow/cancellation of squared tiny moments.
    u=math.exp(l7-l5);v=math.exp(l9-l5)
    n=28+80*v-64*u*u;d=77-112*u+80*v
    dn=80*v*(s9-s5)-128*u*u*(s7-s5)
    dd=-112*u*(s7-s5)+80*v*(s9-s5)
    ks=k5*(n/d)*(45/44)
    return k5,ks,s5,s5+dn/n-dd/d

@njit(cache=True)
def conductivity(T,rho,sigmahat,what,alph,a,b,c,order=1):
    """Return sqrt(T)/D and its log derivative w.r.t. T at fixed rho."""
    kl,ks,sl,ss=factors(T,what,order)
    ls=math.log(a/b)+2*math.log(sigmahat)+math.log(ks)
    ll=-math.log(c*rho*T*kl)
    z=alph*(ls-ll);ws=math.exp(-softplus(-z));wl=1-ws
    ld=max(ls,ll)+math.log1p(math.exp(-alph*abs(ls-ll)))/alph
    k=math.exp(.5*math.log(T)-ld)
    return k,.5-ws*ss+wl*(1+sl)
