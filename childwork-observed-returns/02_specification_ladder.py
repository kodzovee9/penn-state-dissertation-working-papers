"""
Specification ladder in economy E3 (Figure 3), computed WITHOUT measurement
noise at N = 200,000 so that the full specification recovers the model exactly:
(0.0000, alpha = 0.5000, kappa = 1.0000). Produces specification_ladder.png.
Run after 01_population_economies.py if regenerating all figures (that script
saves its noisy ladder variant under a different name).
"""
import numpy as np

P = dict(alpha=0.5, w=0.3, w2=3.5, theta=0.25, R=1.4, beta=0.8)
LMAX = 0.92

def solve_economy(a, A, sigma):
    alpha, w, w2, theta, R, beta = P['alpha'], P['w'], P['w2'], P['theta'], P['R'], P['beta']
    w1 = np.full_like(a, w)
    s_u = (R*(w1+theta)/(w2*a*alpha))**(1.0/(alpha-1.0))
    l_u = np.clip(1.0-s_u, 0.0, LMAX); s_u = 1.0-l_u
    kap = (beta*R)**(-1.0/sigma)
    d_u = (kap*w2*a*s_u**alpha - (A+(w1+theta)*l_u-theta))/(1.0+kap*R)
    unc = d_u <= 0.0
    l = np.where(unc, l_u, np.nan); con = ~unc
    if con.any():
        ac, Ac, w1c = a[con], A[con], w1[con]
        def G(lc):
            sc = 1.0-lc; c1 = Ac+(w1c+theta)*lc-theta
            with np.errstate(invalid='ignore'):
                return (w1c+theta)*np.where(c1>0,c1,np.nan)**(-sigma) \
                    - beta*w2*ac*alpha*sc**(alpha-1.0)*(w2*ac*sc**alpha)**(-sigma)
        lo = np.maximum(1e-9,(theta-Ac)/(w1c+theta)+1e-9)
        hi = np.full_like(ac, LMAX)
        c0 = (lo<=1e-8)&(G(np.maximum(np.full_like(ac,1e-9),lo))<=0)
        cT = G(hi)>0
        lom,him = lo.copy(),hi.copy()
        for _ in range(80):
            mid=0.5*(lom+him); pos=G(mid)>0
            lom=np.where(pos,mid,lom); him=np.where(pos,him,mid)
        l[con]=np.where(c0,0.0,np.where(cT,hi,0.5*(lom+him)))
    return l

def ols(y,X):
    X1=np.column_stack([np.ones(len(y))]+[x for x in X])
    return np.linalg.lstsq(X1,y,rcond=None)[0][1:]

if __name__ == "__main__":
    rng=np.random.default_rng(42); N=200000
    lna=rng.normal(0,0.35,N); a=np.exp(lna)
    A=np.exp(rng.normal(0.7,0.15,N))
    l=solve_economy(a,A,3.0)
    s=1.0-l; lny=np.log(P['w2']*a*s**0.5); lns=np.log(s)   # no noise
    vals=dict(naive=ols(lny,[l])[0], sel=ols(lny,[l,lns])[0], ctrl=ols(lny,[l,lna])[0])
    bf=ols(lny,[l,lns,lna]); vals['full_l'],vals['full_s'],vals['full_a']=bf
    print(f"naive={vals['naive']:+.3f} RS(l)={vals['sel']:+.3f} RA(l)={vals['ctrl']:+.3f} "
          f"full=({vals['full_l']:+.4f},{vals['full_s']:.4f},{vals['full_a']:.4f})")
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(7.4,4.5))
    labels=['naive:\nln y on l','+ schooling:\ncoef on l','+ ability:\ncoef on l','full model:\ncoef on l']
    v=[vals['naive'],vals['sel'],vals['ctrl'],vals['full_l']]
    cols=['#c0504d' if x>0.005 else '#4472c4' for x in v]
    ax.bar(labels,v,color=cols,width=0.55)
    ax.axhline(0,color='k',lw=0.9)
    for i,x in enumerate(v):
        ax.text(i,x+(0.03 if x>=0 else -0.07),f'{x:+.2f}',ha='center',fontsize=12)
    ax.set_ylabel('coefficient on child work $l$')
    ax.set_title('Specification ladder, economy E3\n(no measurement noise: the full specification recovers the model exactly)')
    fig.tight_layout()
    fig.savefig('specification_ladder.png',dpi=150,bbox_inches='tight')
    print("figure saved: specification_ladder.png")
