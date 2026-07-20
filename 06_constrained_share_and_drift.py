"""
(A) Credit-expansion drift in a NEGATIVE-selection economy (E1, sigma = 0.7):
the naive coefficient moves down, so the policy-induced drift direction
distinguishes the selection regimes (paper, Section 5 and Appendix B).
(B) Constrained-share robustness (Figure 7): positive naive coefficients do
not require near-universal credit exclusion; the positive region is interior
in constraint prevalence.
kappa = 1, delta = 0 throughout.
"""

import numpy as np

BASE = dict(alpha=0.5, w=0.3, w2=3.5, theta=0.25, R=1.4, beta=0.8)
LMAX = 0.92


def solve_economy(a, A, sigma, dbar, p):
    alpha, w, w2, theta, R, beta = p['alpha'], p['w'], p['w2'], p['theta'], p['R'], p['beta']
    w1 = np.full_like(a, w)
    s_u = (R * (w1 + theta) / (w2 * a * alpha))**(1.0 / (alpha - 1.0))
    l_u = np.clip(1.0 - s_u, 0.0, LMAX)
    s_u = 1.0 - l_u
    kap = (beta * R)**(-1.0 / sigma)
    d_u = (kap * w2 * a * s_u**alpha - (A + (w1 + theta) * l_u - theta)) / (1.0 + kap * R)
    unc = d_u <= dbar
    l = np.where(unc, l_u, np.nan)
    con = ~unc
    if con.any():
        ac, Ac, w1c = a[con], A[con], w1[con]

        def G(lc):
            sc = 1.0 - lc
            c1 = Ac + (w1c + theta) * lc - theta + dbar
            c2 = w2 * ac * sc**alpha - R * dbar
            m = ac * alpha * sc**(alpha - 1.0)
            with np.errstate(invalid='ignore'):
                return (w1c + theta) * np.where(c1 > 0, c1, np.nan)**(-sigma) \
                    - beta * w2 * m * np.where(c2 > 0, c2, np.nan)**(-sigma)

        lo = np.maximum(1e-9, (theta - Ac - dbar) / (w1c + theta) + 1e-9)
        hi = np.full_like(ac, LMAX)
        if dbar > 0:
            s_min = (R * dbar / (w2 * ac))**(1.0 / alpha)
            hi = np.minimum(hi, 1.0 - np.minimum(s_min * 1.02 + 1e-6, 0.999))
        corner0 = (lo <= 1e-8) & (G(np.maximum(np.full_like(ac, 1e-9), lo)) <= 0)
        cornerT = G(hi) > 0
        lom, him = lo.copy(), hi.copy()
        for _ in range(80):
            mid = 0.5 * (lom + him)
            pos = G(mid) > 0
            lom = np.where(pos, mid, lom)
            him = np.where(pos, him, mid)
        sol = np.where(corner0, 0.0, np.where(cornerT, hi, 0.5 * (lom + him)))
        l[con] = sol
    return l, con


def ols(y, X):
    X1 = np.column_stack([np.ones(len(y))] + [x for x in X])
    return np.linalg.lstsq(X1, y, rcond=None)[0][1:]


def run(N, sigma, mu_A, s_A, dbar, seed=42, noise=0.3):
    rng = np.random.default_rng(seed)
    lna = rng.normal(0.0, 0.35, N)
    a = np.exp(lna)
    A = np.exp(rng.normal(mu_A, s_A, N))
    l, con = solve_economy(a, A, sigma, dbar, BASE)
    h = a * (1.0 - l)**BASE['alpha']
    lny = np.log(BASE['w2'] * h) + rng.normal(0.0, noise, N)
    return dict(con=con.mean(), mean_l=l.mean(), b=ols(lny, [l])[0])


# ---- A: drift direction in the negative-selection economy E1 ----
print("=" * 76)
print("A. Credit expansion in E1 (sigma=0.7, negative selection): drift direction")
print("=" * 76)
print(f"{'dbar':>6}{'con%':>7}{'mean l':>9}{'b_naive':>10}")
e1 = []
for dbar in [0.0, 0.1, 0.2, 0.4, 0.6]:
    o = run(20000, 0.7, 0.7, 0.35, dbar)
    e1.append((dbar, o))
    print(f"{dbar:>6.1f}{o['con']:>7.0%}{o['mean_l']:>9.3f}{o['b']:>10.3f}")

# ---- B: constrained-share robustness (sigma=3) ----
print()
print("=" * 76)
print("B. beta_naive vs constrained share (sigma=3), three ways of varying the share")
print("=" * 76)
series = {}
series['mean wealth ($s_A{=}0.35$)'] = [run(12000, 3.0, muA, 0.35, 0.0, seed=11)
                                        for muA in np.linspace(0.2, 1.8, 9)]
series['mean wealth ($s_A{=}0.15$)'] = [run(12000, 3.0, muA, 0.15, 0.0, seed=11)
                                        for muA in np.linspace(0.2, 1.8, 9)]
series['borrowing limit ($\\mu_A{=}0.7$)'] = [run(12000, 3.0, 0.7, 0.35, db, seed=11)
                                              for db in np.linspace(0.0, 0.6, 7)]
pos_shares = []
for name, rows in series.items():
    print(f"-- {name}")
    for o in rows:
        flag = " <-- beta > 0" if o['b'] > 0 else ""
        print(f"   con={o['con']:>4.0%}  mean_l={o['mean_l']:.3f}  b={o['b']:+.3f}{flag}")
        if o['b'] > 0:
            pos_shares.append(o['con'])
print(f"\nconstrained shares with beta_naive > 0: min={min(pos_shares):.0%}, max={max(pos_shares):.0%}")

# ---- figure ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(figsize=(7.6, 4.8))
styles = [('o-', '#4472c4'), ('s--', '#c0504d'), ('d-.', '#e8862d')]
for (name, rows), (mk, col) in zip(series.items(), styles):
    xs = [o['con'] for o in rows]
    ys = [o['b'] for o in rows]
    order = np.argsort(xs)
    ax.plot(np.array(xs)[order], np.array(ys)[order], mk, color=col, label=name, lw=2)
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('constrained population share')
ax.set_ylabel(r'$\beta_{naive}$')
ax.set_title(r'Positive coefficients do not require universal credit exclusion ($\sigma=3$)')
ax.legend(frameon=False, fontsize=11)
fig.tight_layout()
fig.savefig('conshare_robustness.png', dpi=150, bbox_inches='tight')
print("figure saved: conshare_robustness.png")
