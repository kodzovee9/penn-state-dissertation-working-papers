"""
Skill-building channel robustness (delta > 0), paper Section 4.5.

h = a s^alpha + delta * l (eta = 1). Verifies the interior-margin lemma
(m > 0 at every interior optimum, any delta), the constrained selection
threshold with delta > 0, the showcase economies at delta in {0, 0.1, 0.2},
and the sign-flip sweep at delta = 0.2.
"""

import numpy as np

P = dict(alpha=0.5, w=0.3, w2=3.5, theta=0.25, R=1.4, beta=0.8)
LMAX = 0.92
S_A = 0.35


def solve_economy(a, A, sigma, phi, delta, p=P, lmax=LMAX):
    alpha, w, w2, theta, R, beta = p['alpha'], p['w'], p['w2'], p['theta'], p['R'], p['beta']
    w1 = w * a**phi

    # unconstrained: a alpha s^(alpha-1) = R(w1+theta)/w2 + delta  (closed form, eta=1)
    rhs = R * (w1 + theta) / w2 + delta
    s_u = (rhs / (a * alpha))**(1.0 / (alpha - 1.0))
    l_u = np.clip(1.0 - s_u, 0.0, lmax)
    s_u = 1.0 - l_u
    H_u = a * s_u**alpha + delta * l_u
    kappa = (beta * R)**(-1.0 / sigma)
    d_u = (kappa * w2 * H_u - (A + (w1 + theta) * l_u - theta)) / (1.0 + kappa * R)
    unc = d_u <= 0.0

    l = np.where(unc, l_u, np.nan)
    con = ~unc
    if con.any():
        ac, Ac, w1c = a[con], A[con], w1[con]

        def hfun(lc):
            return ac * (1.0 - lc)**alpha + delta * lc

        def G(lc):
            sc = 1.0 - lc
            c1 = Ac + (w1c + theta) * lc - theta
            m = ac * alpha * sc**(alpha - 1.0) - delta
            c2 = w2 * hfun(lc)
            with np.errstate(invalid='ignore'):
                return (w1c + theta) * np.where(c1 > 0, c1, np.nan)**(-sigma) \
                    - beta * w2 * m * c2**(-sigma)

        lo = np.maximum(1e-9, (theta - Ac) / (w1c + theta) + 1e-9)
        hi = np.full_like(ac, lmax)
        corner0 = (lo <= 1e-8) & (G(np.maximum(np.full_like(ac, 1e-9), lo)) <= 0)
        cornerT = G(hi) > 0
        lom, him = lo.copy(), hi.copy()
        for _ in range(80):
            mid = 0.5 * (lom + him)
            pos = G(mid) > 0
            lom = np.where(pos, mid, lom)
            him = np.where(pos, him, mid)
        sol = 0.5 * (lom + him)
        sol = np.where(corner0, 0.0, np.where(cornerT, lmax, sol))
        l[con] = sol
    return l, con


def ols(y, X):
    X1 = np.column_stack([np.ones(len(y))] + [x for x in X])
    return np.linalg.lstsq(X1, y, rcond=None)[0][1:]


def run_economy(N, sigma, phi, mu_A, s_A, delta, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    lna = rng.normal(0.0, S_A, N)
    a = np.exp(lna)
    A = np.exp(rng.normal(mu_A, s_A, N))
    l, con = solve_economy(a, A, sigma, phi, delta)
    s = 1.0 - l
    h = a * s**P['alpha'] + delta * l
    m = a * P['alpha'] * s**(P['alpha'] - 1.0) - delta
    lny2 = np.log(P['w2'] * h) + rng.normal(0.0, noise, N)

    corner0 = l <= 1e-7
    interior = ~corner0 & (l < LMAX - 1e-7)
    work = l > 1e-7

    out = dict(
        con_share=con.mean(), l0=corner0.mean(),
        lmean_w=l[work].mean() if work.any() else 0.0,
        min_m_interior=m[interior].min() if interior.any() else np.nan,
        n_corner0_mneg=int((corner0 & (a * P['alpha'] - delta <= 0)).sum()),
        avg_marg_eff=(-m[work] / h[work]).mean() if work.any() else np.nan,
        b_naive=ols(lny2, [l])[0],
        b_ctrl=ols(lny2, [l, lna])[0],
    )
    # within-constrained selection slope (interior only)
    cint = con & interior
    out['sel_con'] = (np.cov(lna[cint], l[cint])[0, 1] / np.var(lna[cint])) if cint.sum() > 50 else np.nan
    return out


# ---------------- C1 + C2: theorem checks ----------------

print("=" * 96)
print("C1. Interior-optimum theorem: min m across interior optima (must be > 0), and")
print("    no l=0 corners among households with m(0) = a*alpha - delta <= 0")
print("=" * 96)
rng = np.random.default_rng(5)
worst_m, viol = np.inf, 0
for _ in range(60):
    sig = float(rng.uniform(0.5, 5.0)); phi = float(rng.uniform(0, 0.8))
    dlt = float(rng.uniform(0.0, 0.45)); muA = float(rng.uniform(0.2, 1.4))
    o = run_economy(3000, sig, phi, muA, 0.35, dlt, seed=int(rng.integers(1e6)))
    if np.isfinite(o['min_m_interior']):
        worst_m = min(worst_m, o['min_m_interior'])
    viol += o['n_corner0_mneg']
print(f"60 random economies (delta up to 0.45): min interior m = {worst_m:.4f}  (>0 required)")
print(f"l=0 corners among m(0)<=0 households: {viol}  (0 required)")

print()
print("=" * 96)
print("C2. Constrained selection threshold with delta>0 (spot check, finite differences)")
print("=" * 96)
from scipy.optimize import brentq
def single(a, A, sigma, delta, p=P):
    l, con = solve_economy(np.array([a]), np.array([A]), sigma, 0.0, delta)
    return float(l[0]), bool(con[0])
ok = tot = 0
rng = np.random.default_rng(9)
for _ in range(400):
    sig = float(rng.uniform(0.5, 5.0)); dlt = float(rng.uniform(0.0, 0.3))
    a = float(rng.uniform(0.7, 1.5)); A = float(rng.uniform(0.5, 2.5))
    l0, con0 = single(a, A, sig, dlt)
    if not con0 or l0 <= 1e-6 or l0 >= LMAX - 1e-6:
        continue
    lp, cp = single(a + 1e-5, A, sig, dlt); lm, cm = single(a - 1e-5, A, sig, dlt)
    if not (cp and cm):
        continue
    dlda = (lp - lm) / 2e-5
    s = 1 - l0
    m = a * P['alpha'] * s**(P['alpha'] - 1) - dlt
    h = a * s**P['alpha'] + dlt * l0
    c2 = P['w2'] * h
    sigstar = P['alpha'] * s**(P['alpha'] - 1) * c2 / (P['w2'] * m * s**P['alpha'])
    tot += 1
    if ((sig > sigstar) == (dlda > 0)) or abs(sig - sigstar) < 2e-2:
        ok += 1
print(f"threshold sign agreement: {ok}/{tot}")

# ---------------- C3: E1-E4 across delta ----------------

print()
print("=" * 96)
print("C3. Showcase economies across delta (N=20,000)")
print("=" * 96)
cases = [("E1 (sig=0.7, sA=.35)", 0.7, 0.0, 0.7, 0.35),
         ("E2 (sig=3, sA=.35)", 3.0, 0.0, 0.7, 0.35),
         ("E3 (sig=3, sA=.15)", 3.0, 0.0, 0.7, 0.15),
         ("E4 (sig=3, phi=.5, sA=.15)", 3.0, 0.5, 0.7, 0.15)]
print(f"{'economy':<30}{'delta':>6}{'con%':>6}{'l=0%':>6}{'selcon':>8}{'naive':>8}{'ctrl(a)':>9}{'E[-m/h]':>9}")
for name, sig, phi, muA, sA in cases:
    for dlt in [0.0, 0.1, 0.2]:
        o = run_economy(20000, sig, phi, muA, sA, dlt, seed=42)
        print(f"{name:<30}{dlt:>6.1f}{o['con_share']:>6.0%}{o['l0']:>6.0%}"
              f"{o['sel_con']:>8.3f}{o['b_naive']:>8.3f}{o['b_ctrl']:>9.3f}{o['avg_marg_eff']:>9.3f}")

# ---------------- C4: sign-flip sweep at delta=0.2 ----------------

print()
print("=" * 96)
print("C4. Sign-flip sweep at delta=0.2: beta_naive vs sd(lnA)  (sigma=3, phi=0)")
print("=" * 96)
for sA in [0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00]:
    o = run_economy(8000, 3.0, 0.0, 0.7, sA, 0.2, seed=11)
    print(f"  sd(lnA)={sA:.2f}: beta_naive={o['b_naive']:+.3f}   ctrl(a)={o['b_ctrl']:+.3f}   E[-m/h]={o['avg_marg_eff']:+.3f}")
