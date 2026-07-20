"""
Exact constrained reversal boundary (delta = 0), Proposition 3(ii).

Reversal at a constrained interior optimum iff
    phi*w1/q  >  kappa/alpha + (sigma/c1) * [ phi*w1*l + kappa*(1-l)*q/alpha ],
with q = w1 + theta. The unconstrained boundary is phi*w1/q > kappa/alpha;
the additional strictly positive term shows credit constraints raise the
reversal threshold. Reproduces the illustrative constrained-reversal example
(phi/kappa ~ 10.8) reported in Appendix B, and verifies the boundary against
a broad random scan (zero sign mismatches). Derivation: paper, Appendix A.
"""
import numpy as np
from scipy.optimize import brentq

def solve_con(a, A, sigma, phi, kappa, dbar, p):
    al, w, w2, th, R, be = p['alpha'], p['w'], p['w2'], p['theta'], p['R'], p['beta']
    w1 = w * a**phi
    def G(l):
        s = 1-l; c1 = A + (w1+th)*l - th + dbar
        c2 = w2 * a**kappa * s**al - R*dbar
        m = a**kappa * al * s**(al-1)
        return (w1+th)*c1**(-sigma) - be*w2*m*c2**(-sigma)
    lo = max(1e-9, (th - A - dbar)/(w1+th) + 1e-9)
    hi = 0.999
    while hi > lo and p['w2']*a**kappa*(1-hi)**al - R*dbar <= 0: hi -= 1e-3
    if G(lo+1e-12) < 0 or G(hi) > 0: return None
    return brentq(G, lo+1e-12, hi, xtol=1e-14)

if __name__ == "__main__":
    # counterexample (phi/kappa ~ 10.8)
    a0 = 1.058
    p = dict(alpha=0.688, w=2.251/a0, w2=11.928, theta=0.009, R=1.313, beta=0.357)
    sigma, phi, kappa, dbar, A = 0.893, 1.0, 0.093, 0.793, 0.0075
    f = lambda aa: solve_con(aa, A, sigma, phi, kappa, dbar, p)
    l0 = f(a0); ha = 1e-6
    lp, lm = f(a0+ha), f(a0-ha)
    hfun = lambda aa, ll: aa**kappa*(1-ll)**p['alpha']
    print("l =", round(l0, 4), " dl/da =", round((lp-lm)/(2*ha), 4),
          " dh/da =", round((hfun(a0+ha, lp)-hfun(a0-ha, lm))/(2*ha), 5))

    # scan: exact boundary vs finite differences
    rng = np.random.default_rng(7); mis = tot = nrev = 0
    for _ in range(4000):
        al_ = float(rng.uniform(0.3, 0.8)); kap_ = float(rng.uniform(0.05, 1.0))
        phi_ = float(rng.uniform(0.0, 1.0)); sig_ = float(rng.uniform(0.3, 4.0))
        p2 = dict(alpha=al_, w=float(rng.uniform(0.2, 2.5)), w2=float(rng.uniform(1.5, 12.0)),
                  theta=float(rng.uniform(0.005, 0.4)), R=float(rng.uniform(1.1, 1.9)),
                  beta=float(rng.uniform(0.3, 0.9)))
        db_ = float(rng.uniform(0.0, 0.8)); A_ = float(rng.uniform(0.005, 1.5)); a_ = float(rng.uniform(0.6, 1.8))
        g = lambda aa: solve_con(aa, A_, sig_, phi_, kap_, db_, p2)
        try: l0 = g(a_)
        except Exception: continue
        if l0 is None or not (1e-4 < l0 < 0.99-1e-4): continue
        w1_ = p2['w']*a_**phi_; q_ = w1_+p2['theta']; s_ = 1-l0
        c1_ = A_ + q_*l0 - p2['theta'] + db_
        c2_ = p2['w2']*a_**kap_*s_**al_ - p2['R']*db_
        if c1_ <= 0 or c2_ <= 0 or c1_**(-sig_) - p2['beta']*p2['R']*c2_**(-sig_) <= 0: continue
        try: lp, lm = g(a_+1e-6), g(a_-1e-6)
        except Exception: continue
        if lp is None or lm is None: continue
        dh = ((a_+1e-6)**kap_*(1-lp)**al_ - (a_-1e-6)**kap_*(1-lm)**al_)/2e-6
        lhs_ = phi_*w1_/q_; rhs_ = kap_/al_ + (sig_/c1_)*(phi_*w1_*l0 + kap_*s_*q_/al_)
        tot += 1; nrev += dh < 0
        if (lhs_ > rhs_) != (dh < 0) and abs(lhs_-rhs_) > 1e-3: mis += 1
    print(f"scan: {tot} optima | reversals: {nrev} | mismatches: {mis}")
